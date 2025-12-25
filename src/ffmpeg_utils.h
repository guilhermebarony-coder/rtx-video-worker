#pragma once

extern "C"
{
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/avutil.h>
}

#include <stdexcept>
#include <string>

#include "pipeline_types.h"
#include "audio_config.h"

// Forward declarations
struct PipelineConfig;

inline void ff_check(int err, const char *what)
{
    if (err < 0)
    {
        char buf[256];
        av_strerror(err, buf, sizeof(buf));
        throw std::runtime_error(std::string(what) + ": " + buf);
    }
}

inline std::string ff_ts(double seconds)
{
    char b[64];
    snprintf(b, sizeof(b), "%.3fs", seconds);
    return b;
}

// Ensure DTS monotonicity for muxer compatibility (MP4 requires strictly increasing DTS)
// Updates packet DTS if needed and ensures PTS >= DTS. Updates last_dts tracking variable.
inline void ensure_dts_monotonicity(AVPacket *pkt, int64_t &last_dts)
{
    if (last_dts != AV_NOPTS_VALUE && pkt->dts != AV_NOPTS_VALUE)
    {
        int64_t min_dts = last_dts + 1;
        if (pkt->dts < min_dts)
        {
            pkt->dts = min_dts;
            // Ensure PTS >= DTS after adjustment
            if (pkt->pts != AV_NOPTS_VALUE && pkt->pts < pkt->dts)
            {
                pkt->pts = pkt->dts;
            }
        }
    }
    last_dts = pkt->dts;
}

// Attach HDR mastering metadata and content light level side data to a video stream
// Used for SDR→HDR (THDR) to generate synthetic metadata
void add_mastering_and_cll(AVStream *st, int max_luminance_nits);

// Read and preserve HDR metadata from input stream to output stream
// Used for HDR→HDR passthrough to maintain original mastering display characteristics
bool copy_stream_hdr_metadata(const AVStream *input_stream, AVStream *output_stream);

// Copy per-frame HDR side data (supports HDR10+, Dolby Vision, etc.)
// Call after frame processing to preserve dynamic metadata
void copy_frame_hdr_side_data(const AVFrame *src_frame, AVFrame *dst_frame);

// Open input and locate video stream, prepare decoder. Tries to enable CUDA device.
bool open_input(const char *inPath, InputContext &in, const InputOpenOptions *options = nullptr);
void close_input(InputContext &in);

// Open multiple inputs (for multi-input support)
bool open_inputs(const std::vector<std::string> &inPaths, std::vector<InputContext> &inputs, const InputOpenOptions *options = nullptr);
void close_inputs(std::vector<InputContext> &inputs);

// Open output, create video encoder stream and map non-video streams.
bool open_output(const char *outPath, const InputContext &in, OutputContext &out, const std::vector<std::string> &streamMaps = {}, const std::string &outputFormatName = "");
void close_output(OutputContext &out);

// Apply metadata and chapter mapping settings (Jellyfin compatibility)
void apply_metadata_chapter_settings(OutputContext &out, const PipelineConfig &cfg, const InputContext &in);

// Audio configuration functions
void configure_audio_from_params(const AudioParameters &params, OutputContext &out);
bool apply_stream_mappings(const std::vector<std::string> &mappings, const InputContext &in, OutputContext &out);
bool setup_audio_encoders(const InputContext &in, OutputContext &out); // Multi-stream encoder setup
bool setup_audio_decoders(InputContext &in, const OutputContext &out); // Multi-stream decoder setup
bool process_audio_frame_multi(AVFrame *input_frame, int input_stream_index, OutputContext &out); // Multi-stream processing
