#pragma once

extern "C"
{
#include "codecclean.cuh"
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/pixfmt.h>
#include <libavutil/frame.h>
#include <libswscale/swscale.h>
}

#include <memory>

#include "rtx_processor.h"
#include "frame_pool.h"

class IProcessor
{
public:
    virtual ~IProcessor() = default;
    // Produces an encoder-ready frame (GPU: AV_PIX_FMT_CUDA P010; CPU: AV_PIX_FMT_P010LE)
    // Returns false on failure.
    virtual bool process(const AVFrame *decframe, AVFrame *&outFrame) = 0;
    virtual void shutdown() = 0;
    // Filtro com janela (CodecClean). Default: nao ha nada preso.
    virtual void filterMarkEnd() {}
    virtual bool filterDrain(AVFrame *&outFrame) { outFrame = nullptr; return false; }
    virtual bool codecCleanOn() const { return false; }
};

class GpuProcessor : public IProcessor
{
public:
    GpuProcessor(RTXProcessor &rtx, CudaFramePool &pool, AVColorSpace colorSpace, bool thdrEnabled, bool inputIsHDR = false)
        : m_rtx(rtx), m_pool(pool), m_bt2020(colorSpace == AVCOL_SPC_BT2020_NCL), m_thdrEnabled(thdrEnabled), m_inputIsHDR(inputIsHDR) {}

    // CodecClean: filtra o LUMA antes do VSR. Sem blob, nao faz nada e
    // o caminho e o de sempre — bit a bit.
    //
    // A JANELA E DE 7 QUADROS CENTRADA, entao o filtro so pode entregar
    // o quadro i depois de ver o i+3. Isso e LATENCIA DE 3 QUADROS num
    // pipeline que hoje e 1:1 depth 0, e quem chama tem que drenar no
    // fim (`filterDrain`). Sem drenar, o video termina 3 quadros mais
    // cedo — o mesmo modo de falha do iter 92/93, que nao aparece como
    // erro porque o arquivo abre e roda.
    bool enableCodecClean(const std::string &blob, float strength,
                          int w, int h)
    {
        if (blob.empty())
            return true;                    // desligado: nada muda
        // MESMO stream do VSR: ele e non-blocking e nao sincroniza
        // com o stream 0. Rodar o filtro no 0 deixava o VSR ler o
        // buffer enquanto os kernels ainda escreviam.
        if (!m_cc.init(blob.c_str(), w, h, m_rtx.stream()))
            return false;
        m_ccOn = true;
        m_ccK = strength;
        m_ccW = w;
        m_ccH = h;
        if (cudaMalloc(&m_ccOut, (size_t)w * h) != cudaSuccess)
            return false;
        return true;
    }

    bool codecCleanOn() const override { return m_ccOn; }

    // Fim da entrada: os ultimos quadros passam a poder sair com a
    // vizinhanca clampada, como nas pontas do lado de Python.
    void filterMarkEnd() override { if (m_ccOn) m_cc.markEnd(); }

    // Produz o proximo quadro preso na janela. Devolve false quando
    // acabou. **Quem chama TEM que drenar ate false** — senao o video
    // termina 3 quadros mais cedo, e isso nao aparece como erro.
    bool filterDrain(AVFrame *&outFrame) override
    {
        outFrame = nullptr;
        if (!m_ccOn)
            return false;
        // Pergunta ANTES de adquirir: o pool e circular e cada
        // acquire() recicla um slot. Adquirir para so entao
        // descobrir que nao ha saida consumiria um slot a toa.
        if (m_cc.available() <= 0)
            return false;
        AVFrame *enc_hw = m_pool.acquire();
        if (!m_cc.pop(m_ccOut, m_ccW, m_ccK, &m_lastUV, &m_lastUVPitch))
        {
            return false;
        }
        // O drain sai pelo MESMO caminho que o quadro normal sairia:
        // com THDR o destino e P010, sem ele e NV12. Um drain que
        // ignorasse isso entregaria os 3 ultimos quadros por outro
        // caminho que o resto do video.
        const bool okDrain =
            m_thdrEnabled
                ? m_rtx.processGpuNV12ToP010(m_ccOut, m_ccW, m_lastUV,
                                             m_lastUVPitch, enc_hw, m_bt2020)
                : m_rtx.processGpuNV12ToNV12(m_ccOut, m_ccW, m_lastUV,
                                             m_lastUVPitch, enc_hw, m_bt2020);
        if (!okDrain)
        {
            return false;
        }
        outFrame = enc_hw;
        return true;
    }

    bool process(const AVFrame *decframe, AVFrame *&outFrame) override
    {
        if (!decframe || decframe->format != AV_PIX_FMT_CUDA)
            return false;

        // Determine the actual pixel format of the CUDA frame
        AVPixelFormat sw_format = AV_PIX_FMT_NONE;
        if (decframe->hw_frames_ctx)
        {
            AVHWFramesContext *hw_frames_ctx = (AVHWFramesContext *)decframe->hw_frames_ctx->data;
            sw_format = hw_frames_ctx->sw_format;
        }

        // O filtro e 8 BITS por construcao (ring uint8, compose em
        // 0..255). Entrada de 10 bits nao tem como passar por ele —
        // entao ERRO, nao silencio. Seguir sem o filtro entregaria um
        // video silenciosamente diferente do que foi pedido, que e a
        // mesma lei ja aplicada ao blob que nao carrega.
        if (m_ccOn && sw_format == AV_PIX_FMT_P010LE)
        {
            fprintf(stderr, "[codecclean] ERRO: entrada de 10 bits (P010) e o "
                            "filtro opera em 8 bits. Rode sem --cc-blob ou "
                            "converta a entrada para 8 bits.\n");
            return false;
        }

        AVFrame *enc_hw = m_pool.acquire();
        bool ok = false;

        if (sw_format == AV_PIX_FMT_P010LE)
        {
            // P010 input: Determine processing path based on:
            // 1. Transfer characteristic (PQ/HLG) = true HDR
            // 2. THDR enabled = need 10-bit output surface
            // 3. Colorspace (BT.2020) = affects color conversion coefficients

            if (m_inputIsHDR)
            {
                // True HDR input (PQ/HLG transfer): Must preserve 10-bit pipeline
                // THDR must be disabled for HDR input (handled by configure_input_hdr_detection)
                // P010 -> X2BGR10LE (10-bit) -> RTX (VSR with 10-bit source array) -> ABGR10 -> P010
                // The 10-bit source array is allocated when inputIsHDR=true && enableTHDR=false
                ok = m_rtx.processGpuP010ToP010(decframe->data[0], decframe->linesize[0],
                                                decframe->data[1], decframe->linesize[1],
                                                enc_hw, m_bt2020);
            }
            else
            {
                // SDR in 10-bit container (Main10 profile): BT.709/BT.2020 transfer, not PQ/HLG
                // Examples: HEVC Main10 with SDR content, VP9 Profile 2 with SDR
                if (m_thdrEnabled)
                {
                    // THDR enabled: Convert SDR to HDR
                    // P010 -> NV12 (8-bit) -> BGRA8 -> RTX (VSR+THDR) -> ABGR10 -> P010
                    ok = m_rtx.processGpuP010SDRToP010(decframe->data[0], decframe->linesize[0],
                                                       decframe->data[1], decframe->linesize[1],
                                                       enc_hw);
                }
                else
                {
                    // No THDR: Keep as SDR, output NV12 (8-bit)
                    // P010 -> NV12 (8-bit) -> BGRA8 -> RTX (VSR only) -> BGRA8 -> NV12
                    ok = m_rtx.processGpuP010ToNV12(decframe->data[0], decframe->linesize[0],
                                                    decframe->data[1], decframe->linesize[1],
                                                    enc_hw);
                }
            }
        }
        else
        {
            // NV12 input: SDR content
            if (m_thdrEnabled)
            {
                // O THDR muda a SAIDA (10 bits), nao a entrada: aqui ela
                // continua NV12 de 8 bits, entao o filtro roda igual ao
                // ramo SDR. Ate 19/08 este ramo IGNORAVA o filtro em
                // silencio — e como o THDR e o PADRAO do worker, era o
                // caminho mais provavel de o usuario cair.
                const uint8_t *tY = decframe->data[0];
                int tPitch = decframe->linesize[0];
                const uint8_t *tUV = decframe->data[1];
                int tUVPitch = decframe->linesize[1];
                if (m_ccOn)
                {
                    m_cc.push(tY, tPitch, tUV, tUVPitch);
                    if (!m_cc.pop(m_ccOut, m_ccW, m_ccK,
                                  &m_lastUV, &m_lastUVPitch))
                    {
                        outFrame = nullptr;
                        return true;        // enchendo a janela, nao e erro
                    }
                    tY = m_ccOut;
                    tPitch = m_ccW;
                    tUV = m_lastUV;         // croma do MESMO quadro que o luma
                    tUVPitch = m_lastUVPitch;
                }
                ok = m_rtx.processGpuNV12ToP010(tY, tPitch, tUV, tUVPitch,
                                                enc_hw, m_bt2020);
            }
            else
            {
                const uint8_t *srcY = decframe->data[0];
                int srcPitch = decframe->linesize[0];
                if (m_ccOn)
                {
                    // A janela come o quadro e devolve o de 3 atras. Se
                    // ainda nao ha saida (os 3 primeiros), este quadro
                    // NAO produz saida — quem chama trata via
                    // `filterPending`.
                    m_cc.push(srcY, srcPitch,
                              decframe->data[1], decframe->linesize[1]);
                    if (!m_cc.pop(m_ccOut, m_ccW, m_ccK,
                                  &m_lastUV, &m_lastUVPitch))
                    {
                        outFrame = nullptr;
                        return true;        // sem saida AINDA, nao e erro
                    }
                    srcY = m_ccOut;
                    srcPitch = m_ccW;
                }
                // Com o filtro, o croma vem do MESMO quadro que o luma
                // (o ring devolve o par); sem filtro, do decframe atual.
                const uint8_t *uvSrc = m_ccOn ? m_lastUV : decframe->data[1];
                int uvPitch = m_ccOn ? m_lastUVPitch : decframe->linesize[1];
                ok = m_rtx.processGpuNV12ToNV12(srcY, srcPitch, uvSrc, uvPitch,
                                                enc_hw, m_bt2020);
            }
        }
        if (!ok)
            return false;
        outFrame = enc_hw;
        return true;
    }

    void shutdown() override { /* RTX owned by caller */ }

private:
    RTXProcessor &m_rtx;
    CudaFramePool &m_pool;
    bool m_bt2020 = false;
    bool m_thdrEnabled = false;
    bool m_inputIsHDR = false;
    cc::CodecCleanFilter m_cc;
    bool m_ccOn = false;
    float m_ccK = 1.0f;
    int m_ccW = 0, m_ccH = 0;
    uint8_t *m_ccOut = nullptr;
    const uint8_t *m_lastUV = nullptr;
    int m_lastUVPitch = 0;
};

class CpuProcessor : public IProcessor
{
public:
    CpuProcessor(RTXProcessor &rtx,
                 int srcW, int srcH,
                 int dstW, int dstH)
        : m_rtx(rtx), m_srcW(srcW), m_srcH(srcH), m_dstW(dstW), m_dstH(dstH)
    {
        // Allocate BGRA buffer
        m_bgra.reset(av_frame_alloc());
        m_bgra->format = AV_PIX_FMT_RGBA;
        m_bgra->width = m_srcW;
        m_bgra->height = m_srcH;
        if (av_frame_get_buffer(m_bgra.get(), 32) < 0)
            throw std::runtime_error("CpuProcessor: alloc BGRA failed");

        // Defer building output and sws until setConfig() (needs THDR on/off)
        m_sws_to_yuv = nullptr;
    }

    bool process(const AVFrame *decframe, AVFrame *&outFrame) override
    {
        if (!decframe)
            return false;
        // Build/update sws_to_argb if needed
        if (!m_sws_to_argb || m_last_src_format != decframe->format || m_last_src_w != decframe->width || m_last_src_h != decframe->height)
        {
            if (m_sws_to_argb)
                sws_freeContext(m_sws_to_argb);
            m_sws_to_argb = sws_getContext(
                decframe->width, decframe->height, (AVPixelFormat)decframe->format,
                m_srcW, m_srcH, AV_PIX_FMT_RGBA,
                SWS_BILINEAR, nullptr, nullptr, nullptr);
            if (!m_sws_to_argb)
                return false;
            // Select input colorspace based on decoded frame colorspace
            const int *coeffs = (decframe->colorspace == AVCOL_SPC_BT2020_NCL)
                                    ? sws_getCoefficients(SWS_CS_BT2020)
                                    : sws_getCoefficients(SWS_CS_ITU709);
            int srcRange = (decframe->color_range == AVCOL_RANGE_JPEG) ? 1 : 0;
            sws_setColorspaceDetails(m_sws_to_argb, coeffs, srcRange, coeffs, 1, 0, 1 << 16, 1 << 16);
            m_last_src_format = decframe->format;
            m_last_src_w = decframe->width;
            m_last_src_h = decframe->height;
        }

        const uint8_t *srcData[AV_NUM_DATA_POINTERS] = {decframe->data[0], decframe->data[1], decframe->data[2], decframe->data[3]};
        int srcLines[AV_NUM_DATA_POINTERS] = {decframe->linesize[0], decframe->linesize[1], decframe->linesize[2], decframe->linesize[3]};
        if (av_frame_make_writable(m_bgra.get()) < 0)
            return false;
        sws_scale(m_sws_to_argb, srcData, srcLines, 0, decframe->height, m_bgra->data, m_bgra->linesize);

        // RTX CPU process -> ABGR10
        const uint8_t *rtx_data = nullptr;
        uint32_t rtxW = 0, rtxH = 0;
        size_t rtxPitch = 0;
        if (!m_rtx.process(m_bgra->data[0], (size_t)m_bgra->linesize[0], rtx_data, rtxW, rtxH, rtxPitch))
            return false;
        if (rtxW != (uint32_t)m_dstW || rtxH != (uint32_t)m_dstH)
            return false;

        // Build m_sws_to_yuv if needed based on THDR config and output format
        if (!m_sws_to_yuv)
        {
            AVPixelFormat srcPix = m_rtx_cfg.enableTHDR ? AV_PIX_FMT_X2BGR10LE : AV_PIX_FMT_BGRA;
            AVPixelFormat dstPix = m_rtx_cfg.enableTHDR ? AV_PIX_FMT_P010LE : AV_PIX_FMT_NV12;
            m_sws_to_yuv = sws_getContext(
                m_dstW, m_dstH, srcPix,
                m_dstW, m_dstH, dstPix,
                SWS_BILINEAR, nullptr, nullptr, nullptr);
            if (!m_sws_to_yuv)
                return false;
            const int *coeffs = m_rtx_cfg.enableTHDR ? sws_getCoefficients(SWS_CS_BT2020)
                                                     : sws_getCoefficients(SWS_CS_ITU709);
            // Source is RGB(A) from RTX (full-range); destination YUV should be limited-range
            sws_setColorspaceDetails(m_sws_to_yuv,
                                     coeffs, 1,
                                     coeffs, 0,
                                     0, 1 << 16, 1 << 16);
        }

        // Ensure output buffer is allocated and writable
        if (!m_out)
            return false;
        if (av_frame_make_writable(m_out.get()) < 0)
            return false;
        const uint8_t *in_planes[1] = {rtx_data};
        int in_lines[1] = {static_cast<int>(rtxPitch)};
        sws_scale(m_sws_to_yuv, in_planes, in_lines, 0, m_dstH, m_out->data, m_out->linesize);

        outFrame = m_out.get();
        return true;
    }

    void setConfig(const RTXProcessConfig &cfg)
    {
        m_rtx_cfg = cfg;
        // Rebuild output frame and sws to match THDR on/off
        if (m_sws_to_yuv)
        {
            sws_freeContext(m_sws_to_yuv);
            m_sws_to_yuv = nullptr;
        }
        // Allocate output frame in requested format
        m_out.reset(av_frame_alloc());
        if (!m_out)
            throw std::runtime_error("CpuProcessor: alloc out frame failed");
        m_out->format = m_rtx_cfg.enableTHDR ? AV_PIX_FMT_P010LE : AV_PIX_FMT_NV12;
        m_out->width = m_dstW;
        m_out->height = m_dstH;
        if (av_frame_get_buffer(m_out.get(), 32) < 0)
            throw std::runtime_error("CpuProcessor: alloc out buffer failed");

        AVPixelFormat srcPix = m_rtx_cfg.enableTHDR ? AV_PIX_FMT_X2BGR10LE : AV_PIX_FMT_BGRA;
        AVPixelFormat dstPix = m_rtx_cfg.enableTHDR ? AV_PIX_FMT_P010LE : AV_PIX_FMT_NV12;
        m_sws_to_yuv = sws_getContext(
            m_dstW, m_dstH, srcPix,
            m_dstW, m_dstH, dstPix,
            SWS_BILINEAR, nullptr, nullptr, nullptr);
        if (!m_sws_to_yuv)
            throw std::runtime_error("CpuProcessor: sws_to_yuv alloc failed in setConfig");
        const int *coeffs = m_rtx_cfg.enableTHDR ? sws_getCoefficients(SWS_CS_BT2020)
                                                 : sws_getCoefficients(SWS_CS_ITU709);
        // Source is RGB(A) from RTX (full-range); destination YUV should be limited-range
        sws_setColorspaceDetails(m_sws_to_yuv,
                                 coeffs, 1,
                                 coeffs, 0,
                                 0, 1 << 16, 1 << 16);
    }

    void shutdown() override
    {
        if (m_sws_to_argb)
        {
            sws_freeContext(m_sws_to_argb);
            m_sws_to_argb = nullptr;
        }
        if (m_sws_to_yuv)
        {
            sws_freeContext(m_sws_to_yuv);
            m_sws_to_yuv = nullptr;
        }
    }

private:
    RTXProcessor &m_rtx;
    RTXProcessConfig m_rtx_cfg{};
    int m_srcW = 0, m_srcH = 0;
    int m_dstW = 0, m_dstH = 0;

    SwsContext *m_sws_to_argb = nullptr;
    SwsContext *m_sws_to_yuv = nullptr; // to P010 (HDR) or NV12 (SDR)
    int m_last_src_format = AV_PIX_FMT_NONE;
    int m_last_src_w = 0, m_last_src_h = 0;

    FramePtr m_bgra{nullptr, &av_frame_free_single_fp};
    FramePtr m_out{nullptr, &av_frame_free_single_fp};
};
