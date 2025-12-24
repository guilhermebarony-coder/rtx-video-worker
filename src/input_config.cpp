#include "input_config.h"
#include "logger.h"
#include "audio_config.h"

#include <cstring>

extern "C"
{
#include <libavutil/avutil.h>
#include <libavutil/rational.h>
}

// Check if input is a network URL
bool is_network_input(const char *input)
{
    if (!input)
        return false;
    return (std::strncmp(input, "http://", 7) == 0 ||
            std::strncmp(input, "https://", 8) == 0 ||
            std::strncmp(input, "rtmp://", 7) == 0 ||
            std::strncmp(input, "rtsp://", 7) == 0 ||
            std::strncmp(input, "tcp://", 6) == 0 ||
            std::strncmp(input, "udp://", 6) == 0);
}

// Configure input HDR detection and reopen with P010 if needed
bool configure_input_hdr_detection(PipelineConfig &cfg, InputContext &in)
{
    // Detect HDR content and disable THDR if input is already HDR
    bool inputIsHDR = false;
    if (in.vst && in.vst->codecpar)
    {
        AVColorTransferCharacteristic trc = in.vst->codecpar->color_trc;
        AVColorPrimaries primaries = in.vst->codecpar->color_primaries;

        // Log what we found for debugging
        LOG_DEBUG("Input stream color properties: trc=%d, primaries=%d, colorspace=%d",
                  trc, primaries, in.vst->codecpar->color_space);

        // Primary detection: transfer characteristic (most reliable for HDR)
        inputIsHDR = (trc == AVCOL_TRC_SMPTE2084) ||  // PQ (HDR10)
                     (trc == AVCOL_TRC_ARIB_STD_B67); // HLG (Hybrid Log-Gamma)

        // Fallback: if trc is unspecified but primaries indicate HDR
        if (!inputIsHDR && trc == AVCOL_TRC_UNSPECIFIED)
        {
            // BT.2020 primaries often indicate HDR content
            if (primaries == AVCOL_PRI_BT2020)
            {
                LOG_DEBUG("Transfer unspecified but BT.2020 primaries detected, checking for HDR side data...");
                // Don't assume HDR just from primaries - wait for actual frame decoding
            }
        }
    }

    if (inputIsHDR)
    {
        if (cfg.rtxCfg.enableTHDR)
        {
            LOG_INFO("Input content is HDR (transfer characteristic: %s). Disabling THDR to preserve HDR metadata.",
                     in.vst->codecpar->color_trc == AVCOL_TRC_SMPTE2084 ? "PQ/HDR10" : "HLG");
            cfg.rtxCfg.enableTHDR = false;
        }

        // Reopen input with P010 preference for HDR content
        close_input(in);
        InputOpenOptions inputOpts;
        inputOpts.fflags = cfg.fflags;
        inputOpts.preferP010ForHDR = true;
        inputOpts.seekTime = cfg.seekTime;
        inputOpts.noAccurateSeek = cfg.noAccurateSeek;
        inputOpts.seek2any = cfg.seek2any;
        inputOpts.seekTimestamp = cfg.seekTimestamp;
        inputOpts.enableErrorConcealment = !cfg.ffCompatible;
        inputOpts.flushOnSeek = false;
        open_input(cfg.inputPath, in, &inputOpts);
        LOG_INFO("Configured decoder for P010 output to preserve full 10-bit HDR pipeline");
    }

    return inputIsHDR;
}

// Auto-disable VSR for high-resolution inputs
void configure_vsr_auto_disable(PipelineConfig &cfg, const InputContext &in)
{
    if (cfg.rtxCfg.enableVSR)
    {
        bool ge4k = (in.vdec->width > 3840 && in.vdec->height > 2160) ||
                       (in.vdec->width > 2160 && in.vdec->height > 3840);
        if (ge4k)
        {
            LOG_INFO("Input resolution is %dx%d (>=4k). Disabling VSR.", in.vdec->width, in.vdec->height);
            cfg.rtxCfg.enableVSR = false;
        }
    }
}

// Configure audio processing
void configure_audio_processing(PipelineConfig &cfg, InputContext &in, OutputContext &out)
{
    if (cfg.ffCompatible)
    {
        LOG_DEBUG("Compatibility mode enabled, configuring audio...");

        AudioParameters audioParams;
        audioParams.codec = cfg.audioCodec;
        audioParams.channels = cfg.audioChannels;
        audioParams.bitrate = cfg.audioBitrate;
        audioParams.sampleRate = cfg.audioSampleRate;
        audioParams.filter = cfg.audioFilter;
        audioParams.streamMaps = cfg.streamMaps;

        configure_audio_from_params(audioParams, out);
        LOG_DEBUG("Audio config completed, enabled=%s", out.audioConfig.enabled ? "true" : "false");

        if (out.audioConfig.enabled)
        {
            // Stream mappings are now applied in open_output(), no need to call here

            // Skip encoder setup for copy mode (audio will be copied directly)
            if (out.audioConfig.codec == "copy")
            {
                LOG_DEBUG("Audio copy mode enabled, skipping encoder setup");
            }
            else
            {
                // Setup encoders for all streams that need re-encoding
                LOG_DEBUG("Setting up multi-stream audio encoders...");
                if (!setup_audio_encoders(in, out))
                {
                    LOG_WARN("Failed to setup audio encoders, disabling audio processing");
                    out.audioConfig.enabled = false;
                }
                else
                {
                    LOG_DEBUG("Multi-stream audio encoder setup complete");

                    // Setup decoders for all streams marked PROCESS_AUDIO
                    LOG_DEBUG("Setting up audio decoders...");
                    if (!setup_audio_decoders(in, out))
                    {
                        LOG_WARN("Failed to setup audio decoders, disabling audio processing");
                        out.audioConfig.enabled = false;
                    }
                    else
                    {
                        LOG_DEBUG("Audio decoder setup complete");
                    }
                }
            }
        }
    }
    LOG_DEBUG("Audio configuration complete, proceeding...");
}

// Setup progress tracking
int64_t setup_progress_tracking(const InputContext &in, const AVRational &fr)
{
    int64_t total_frames = 0;
    if (in.vst->nb_frames > 0)
    {
        total_frames = in.vst->nb_frames;
    }
    else
    {
        int64_t duration_us = 0;
        if (in.vst->duration > 0 && in.vst->duration != AV_NOPTS_VALUE)
        {
            // Use integer rescaling instead of floating-point to avoid precision loss
            duration_us = av_rescale_q(in.vst->duration, in.vst->time_base, {1, AV_TIME_BASE});
        }
        else if (in.fmt->duration != AV_NOPTS_VALUE)
        {
            duration_us = in.fmt->duration;
        }

        // Account for input seeking: subtract seek offset from total duration
        if (in.seek_offset_us > 0)
        {
            duration_us -= in.seek_offset_us;
            if (duration_us < 0)
            {
                duration_us = 0;
            }
        }

        if (duration_us > 0 && fr.num > 0 && fr.den > 0)
        {
            // Calculate frames using integer rescaling: duration * framerate
            // av_rescale_q(duration_us, {1, AV_TIME_BASE}, frame_rate) gives frame count
            total_frames = av_rescale_q(duration_us, {1, AV_TIME_BASE}, av_inv_q(fr));
        }
    }
    return total_frames;
}
