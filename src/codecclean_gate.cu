// GATE DO PORT: os kernels de codecclean.cu reproduzem a referencia?
//
// Programa autonomo. Le o blob de pesos e o caso de teste que o
// eval/dump_blob.py exportou — as MESMAS janelas e o MESMO residuo
// esperado do golden que ja julgou torch-fp32, torch-fp16, onnx-cpu e
// numpy. Os cinco caminhos passam a ser julgados contra a mesma verdade.
//
// A BARRA E A MESMA do gates_engine.json, escrita antes de existir
// motor: E1 max 0,5 luma, mediana 0,02. Ela nao foi afrouxada para o
// CUDA; se este arquivo nao passar, o problema e este arquivo.
//
// Compilar:
//   nvcc -O2 -arch=sm_120 codecclean.cu codecclean_gate.cu -o ccgate.exe

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>

#include "codecclean.cuh"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA %s @ %d: %s\n", #x, __LINE__, cudaGetErrorString(e_)); \
    exit(1); } } while (0)

static void *rd(FILE *f, size_t n)
{
    void *p = malloc(n);
    if (fread(p, 1, n, f) != n) { fprintf(stderr, "leitura curta\n"); exit(1); }
    return p;
}

int main(int argc, char **argv)
{
    const char *blobPath = argc > 1 ? argv[1] : "cc_32x4.blob";
    const char *casoPath = argc > 2 ? argv[2] : "caso_32x4.bin";

    // ---- blob de pesos -------------------------------------------------
    FILE *fb = fopen(blobPath, "rb");
    if (!fb) { fprintf(stderr, "sem blob: %s\n", blobPath); return 1; }
    char magia[8];
    if (fread(magia, 1, 8, fb) != 8 || memcmp(magia, "CCNET\0\0\0", 8) != 0)
    { fprintf(stderr, "blob invalido\n"); return 1; }
    int hdr[4];
    if (fread(hdr, 4, 4, fb) != 4) return 1;
    const int CH = hdr[0], BL = hdr[1], COND = hdr[2];
    const int CIN = 7 + COND;
    printf("blob: ch=%d blocks=%d cond=%d\n", CH, BL, COND);

    size_t nW = (size_t)CH * CIN * 9 + CH;
    for (int b = 0; b < BL; ++b) nW += 2 * ((size_t)CH * CH * 9 + CH);
    nW += (size_t)CH * 9 + 1;
    float *hW = (float *)rd(fb, nW * 4);
    fclose(fb);

    float *dW; CK(cudaMalloc(&dW, nW * 4));
    CK(cudaMemcpy(dW, hW, nW * 4, cudaMemcpyHostToDevice));

    // offsets, na MESMA ordem em que o dump_blob escreveu
    size_t o = 0;
    float *w_inp = dW + o; o += (size_t)CH * CIN * 9;
    float *b_inp = dW + o; o += CH;
    std::vector<float *> w1(BL), b1(BL), w2(BL), b2(BL);
    for (int b = 0; b < BL; ++b) {
        w1[b] = dW + o; o += (size_t)CH * CH * 9;
        b1[b] = dW + o; o += CH;
        w2[b] = dW + o; o += (size_t)CH * CH * 9;
        b2[b] = dW + o; o += CH;
    }
    float *w_out = dW + o; o += (size_t)CH * 9;
    float *b_out = dW + o;

    // ---- caso de teste --------------------------------------------------
    FILE *fc = fopen(casoPath, "rb");
    if (!fc) { fprintf(stderr, "sem caso: %s\n", casoPath); return 1; }
    int ch4[4];
    if (fread(ch4, 4, 4, fc) != 4) return 1;
    const int NCASO = ch4[0], W = ch4[1], H = ch4[2], SEQ = ch4[3];
    const int N = W * H;
    printf("caso: %d janelas de %d quadros, %dx%d\n\n", NCASO, SEQ, W, H);

    uint8_t *dFrames; CK(cudaMalloc(&dFrames, (size_t)SEQ * N));
    float *dF, *dF2, *dTmp, *dTmp2, *dMaps, *dIn, *dA, *dB, *dC, *dRes;
    CK(cudaMalloc(&dF,   N * 4));
    CK(cudaMalloc(&dF2,  N * 4));
    CK(cudaMalloc(&dTmp, N * 4));
    CK(cudaMalloc(&dTmp2, N * 4));
    CK(cudaMalloc(&dMaps, (size_t)3 * N * 4));
    CK(cudaMalloc(&dIn,  (size_t)CIN * N * 4));
    CK(cudaMalloc(&dA,   (size_t)CH * N * 4));
    CK(cudaMalloc(&dB,   (size_t)CH * N * 4));
    CK(cudaMalloc(&dC,   (size_t)CH * N * 4));
    CK(cudaMalloc(&dRes, N * 4));

    dim3 blk(32, 8), grd((W + 31) / 32, (H + 7) / 8);
    int lin = (N + 255) / 256;

    double piorMax = 0.0, somaMed = 0.0;
    for (int c = 0; c < NCASO; ++c)
    {
        std::vector<uint8_t> frames((size_t)SEQ * N);
        if (fread(frames.data(), 1, frames.size(), fc) != frames.size()) return 1;
        float gop; if (fread(&gop, 4, 1, fc) != 1) return 1;
        std::vector<float> esperado(N);
        if (fread(esperado.data(), 4, N, fc) != (size_t)N) return 1;
        CK(cudaMemcpy(dFrames, frames.data(), frames.size(), cudaMemcpyHostToDevice));

        // --- os 7 quadros normalizados vao para os canais 0..6 -----------
        for (int i = 0; i < SEQ; ++i) {
            cc::k_u8_to_f32<<<grd, blk>>>(dFrames + (size_t)i * N, W,
                                          dIn + (size_t)i * N, nullptr, W, H);
            // a rede recebe 0..1
            cc::k_clamp01_scale<<<lin, 256>>>(dIn + (size_t)i * N, 1.0f / 255.0f, N);
        }

        // --- mapa 1: bloco (do quadro CENTRAL, indice 3) -----------------
        cc::k_u8_to_f32<<<grd, blk>>>(dFrames + (size_t)3 * N, W, dF, dF2, W, H);
        cc::k_seam<<<grd, blk>>>(dF, dTmp, W, H);
        cc::k_dilate3<<<grd, blk>>>(dTmp, dTmp2, W, H);
        cc::k_blur_h<<<grd, blk>>>(dTmp2, dTmp, W, H, 17);
        cc::k_blur_v<<<grd, blk>>>(dTmp, dMaps + 0, W, H, 17);
        cc::k_clamp01_scale<<<lin, 256>>>(dMaps + 0, 1.0f, N);

        // --- mapa 2: flat ------------------------------------------------
        cc::k_box7<<<grd, blk>>>(dF,  dTmp,  W, H);
        cc::k_box7<<<grd, blk>>>(dF2, dTmp2, W, H);
        cc::k_flat<<<lin, 256>>>(dTmp, dTmp2, dMaps + (size_t)N, W, H);

        // --- mapa 3: grao ------------------------------------------------
        // acumula |hf(i) - hf(i-1)| para i = 2,3,4
        float *dAcc = dMaps + (size_t)2 * N;
        CK(cudaMemset(dAcc, 0, N * 4));
        float *dSoma; CK(cudaMalloc(&dSoma, N * 4));
        CK(cudaMemset(dSoma, 0, N * 4));
        for (int i = 2; i <= 4; ++i) {
            for (int par = 0; par < 2; ++par) {
                int idx = i - par;               // primeiro i, depois i-1
                cc::k_u8_to_f32<<<grd, blk>>>(dFrames + (size_t)idx * N, W,
                                              dF, nullptr, W, H);
                cc::k_blur_h<<<grd, blk>>>(dF, dTmp, W, H, 9);
                cc::k_blur_v<<<grd, blk>>>(dTmp, dTmp2, W, H, 9);
                cc::k_grain_acc<<<lin, 256>>>(dF, dTmp2, dAcc, W, H, par == 0);
            }
            cc::k_add<<<lin, 256>>>(dSoma, dAcc, N);
        }
        cc::k_scale<<<lin, 256>>>(dSoma, 1.0f / 3.0f, N);   // sem clamp aqui
        cc::k_blur_h<<<grd, blk>>>(dSoma, dTmp, W, H, 17);
        cc::k_blur_v<<<grd, blk>>>(dTmp, dAcc, W, H, 17);
        cc::k_clamp01_scale<<<lin, 256>>>(dAcc, 1.0f, N);
        CK(cudaFree(dSoma));

        // --- canais 7..9 = mapas, canal 10 = gop -------------------------
        CK(cudaMemcpy(dIn + (size_t)7 * N, dMaps, (size_t)3 * N * 4,
                      cudaMemcpyDeviceToDevice));
        {
            std::vector<float> g(N, gop);
            CK(cudaMemcpy(dIn + (size_t)10 * N, g.data(), N * 4,
                          cudaMemcpyHostToDevice));
        }

        // --- a rede ------------------------------------------------------
        dim3 g3((W + 31) / 32, (H + 7) / 8, CH);
        cc::k_conv3<<<g3, blk>>>(dIn, CIN, dA, CH, w_inp, b_inp, W, H, 1);
        for (int b = 0; b < BL; ++b) {
            // c1 -> relu -> c2, e o resultado SOMA em dA (o residuo do bloco).
            // dB e dC sao buffers distintos: conv3 le e escreve plano a
            // plano, entao src == dst daria corrida entre threads de
            // blocos diferentes — leitura de valor ja sobrescrito.
            cc::k_conv3<<<g3, blk>>>(dA, CH, dB, CH, w1[b], b1[b], W, H, 1);
            cc::k_conv3<<<g3, blk>>>(dB, CH, dC, CH, w2[b], b2[b], W, H, 0);
            cc::k_add<<<(CH * N + 255) / 256, 256>>>(dA, dC, CH * N);
        }
        dim3 g1((W + 31) / 32, (H + 7) / 8, 1);
        cc::k_conv3<<<g1, blk>>>(dA, CH, dRes, 1, w_out, b_out, W, H, 0);
        CK(cudaDeviceSynchronize());

        // --- comparacao --------------------------------------------------
        std::vector<float> got(N);
        CK(cudaMemcpy(got.data(), dRes, N * 4, cudaMemcpyDeviceToHost));
        std::vector<double> d(N);
        for (int i = 0; i < N; ++i) {
            float r = std::max(-1.0f, std::min(1.0f, got[i])) * 255.0f;
            d[i] = std::fabs((double)r - (double)esperado[i]);
        }
        double mx = *std::max_element(d.begin(), d.end());
        std::vector<double> s = d;
        std::nth_element(s.begin(), s.begin() + N / 2, s.end());
        double med = s[N / 2];
        piorMax = std::max(piorMax, mx);
        somaMed += med;
        printf("  janela %d: max %8.5f  mediana %8.5f  (luma)\n", c, mx, med);
    }
    fclose(fc);

    double med = somaMed / NCASO;
    printf("\n%-28s %10s %8s\n", "", "medido", "barra");
    printf("%-28s %10.5f %8.3f\n", "E1 max |d residuo| (luma)", piorMax, 0.5);
    printf("%-28s %10.5f %8.3f\n", "E1 mediana |d| (luma)", med, 0.02);
    bool ok = piorMax <= 0.5 && med <= 0.02;
    printf("\nGATE DO PORT (cuda): %s\n", ok ? "PASSA" : "FALHA");
    return ok ? 0 : 1;
}
