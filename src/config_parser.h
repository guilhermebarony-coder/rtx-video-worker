#pragma once

#include "rtx_processor.h"
#include <string>
#include <vector>
#include <cstdint>

// Configuration structure for the entire processing pipeline
struct PipelineConfig
{
    bool verbose = false;
    bool debug = false;
    bool cpuOnly = false;
    bool ffCompatible = false;

    char *outputPath = nullptr;

    // Multi-input support
    std::vector<std::string> inputPaths;  // Multiple input files (-i flag can be repeated)

    // NVENC settings
    // Preset de qualidade: amarra tune/rc/qp num nome so. Vazio = sem
    // preset (os campos abaixo valem como estao). Ver a resolucao no
    // fim de parse_config, que e onde a regra do cano tambem age.
    std::string quality;
    // Indice absoluto do 1o quadro da entrada (--frame-offset). So o
    // dither do CodecClean usa; existe para preview de um quadro sair
    // bit-identico ao render.
    long long ccFrameOffset = 0;
    // Residuo de media zero (--cc-dc-neutral): tira o deslocamento de
    // brilho do filtro sem encolher a correcao local.
    bool ccDcNeutral = false;
    // Flag explicita ganha do preset: sem estas duas, `--quality master
    // --nvenc-qp 20` daria 12 e o usuario nao entenderia por que.
    bool qpExplicit = false;
    bool tuneExplicit = false;

    std::string tune;
    std::string preset;
    std::string rc; // cbr, vbr, constqp

    int gop; // keyframe interval, multiple of seconds (--nvenc-gop)
    int gopFrames = -1; // GOP size in frames (-g), takes precedence over gop if set
    int bframes;
    int qp;

    // Advanced GOP/keyframe control
    int scThreshold = -1;     // Scene change threshold (-sc_threshold)
    int keyintMin = -1;       // Minimum GOP length (-keyint_min)
    bool noScenecut = false;  // Disable scene detection (-no-scenecut)
    bool forcedIdr = false;   // Force IDR frames (-forced-idr)

    int targetBitrateMultiplier;
    int64_t targetBitrate = -1;     // Target bitrate override (-1 = auto)
    int64_t maxBitrate = -1;        // VBR max bitrate (-1 = 3x target)

    RTXProcessConfig rtxCfg;

    std::string inputFormatName;
    std::string outputFormatName;

    // CodecClean: filtro de residuo de compressao ANTES do VSR.
    // NASCE DESLIGADO — sem --cc-blob o worker se comporta exatamente
    // como antes, bit a bit. O filtro introduz latencia de 3 quadros
    // (janela de 7 centrada), absorvida pelo drain do processor.
    std::string ccBlob;              // caminho dos pesos; vazio = desligado
    float ccStrength = 1.0f;         // k do slider: 0 = bypass exato
    std::string fflags;

    bool overwrite = true;

    // HLS options
    int maxDelay = -1;
    int hlsTime = -1;
    std::string hlsSegmentType;
    std::string hlsInitFilename;
    int64_t hlsStartNumber = -1;
    std::string hlsSegmentFilename;
    std::string hlsPlaylistType;
    int hlsListSize = -1;
    std::string hlsFlags;          // HLS muxer flags (e.g., "independent_segments", "delete_segments")
    std::string hlsSegmentOptions; // Options to pass to segment muxer (e.g., "movflags=+frag_discont")

    // Stream mapping options
    std::vector<std::string> streamMaps;  // Raw -map arguments (e.g., "0:0", "1:a", "-0:v")

    // Stream disable flags (FFmpeg -vn, -an, -sn, -dn)
    bool disableVideo = false;    // -vn: Disable video streams
    bool disableAudio = false;    // -an: Disable audio streams
    bool disableSubtitle = false; // -sn: Disable subtitle streams
    bool disableData = false;     // -dn: Disable data streams

    // Metadata and chapter mapping (Jellyfin compatibility)
    int mapMetadata = 0;          // -map_metadata: -1 = disable, 0+ = input index
    bool hasMapMetadata = false;  // Track if explicitly set (vs default)
    int mapChapters = 0;          // -map_chapters: -1 = disable, 0+ = input index
    bool hasMapChapters = false;  // Track if explicitly set (vs default)

    // Audio codec options
    std::string audioCodec;
    bool audioCodecApplyToAll = false; // true for -codec:a, false for -codec:a:0
    int audioChannels = -1;
    int audioBitrate = -1;
    int audioSampleRate = -1;
    std::string audioFilter;

    // Seek and duration options
    std::string seekTime;        // Input seeking (-ss before -i)
    std::string duration;        // Duration limit (-t)
    std::string outputSeekTime;  // Output seeking (-ss after -i)
    std::string outputTsOffset;  // Output timestamp offset (-output_ts_offset)
    bool copyts = false;         // Preserve original timestamps (-copyts)
    bool noAccurateSeek = false; // Fast seek to nearest keyframe (-noaccurate_seek)
    bool seek2any = false;       // Allow seeking to non-keyframes (-seek2any)
    bool seekTimestamp = false;  // Use timestamp-based seeking (-seek_timestamp)

    // Timestamp handling options (FFmpeg compatibility)
    std::string avoidNegativeTs = "auto"; // FFmpeg -avoid_negative_ts (auto/make_zero/make_non_negative/disabled)
    bool startAtZero = false;             // FFmpeg -start_at_zero
    std::string vsync;                    // FFmpeg -vsync (passthrough/cfr/vfr/drop/auto) - default: auto
    std::string outputFrameRate;          // Output framerate override (-r, -r:v) - overrides input framerate

    // Muxer options (essential - affect output structure/playback)
    std::string movflags;        // User-specified movflags (-movflags)
    int64_t fragDuration = 0;    // Fragment duration in microseconds (-frag_duration)
    int fragmentIndex = -1;      // Fragment index number (-fragment_index)
    int useEditlist = -1;        // Use edit list in MP4 (-use_editlist)
    int maxMuxingQueueSize = -1; // Max muxing queue size (-max_muxing_queue_size)
};

// Print usage/help information
void print_help(const char *argv0);

// Parse command-line arguments and initialize configuration
// Returns true if parsing succeeded, false if help was shown or error occurred
void parse_arguments(int argc, char **argv, PipelineConfig *cfg);
