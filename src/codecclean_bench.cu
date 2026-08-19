// BENCH: onde estao os milissegundos do filtro?
//
// A cadeia num processo dava 55,59 ms/quadro contra 21,9 ms em dois. A
// hipotese era que o `k_conv3` ingenuo fosse o culpado — mas hipotese
// nao e medida. Este arquivo separa os numeros:
//
//   1. o quadro inteiro (push+pop), que e o que o worker paga
//   2. so a REDE, com as formas reais, no caminho ingenuo e no rapido
//   3. uma camada 32->32 sozinha, para ter o custo unitario
//
// A PRIMEIRA MEDIDA (kernel ingenuo, 2026-08-19):
//   quadro 48,96 ms | rede 48,41 ms (98,9%) | resto 0,55 ms
//   2,94 TFLOP/s de ~56 de pico, e so 180 GB/s de 960 de banda.
// Ou seja: nem banda, nem spill — falta de reuso e espera. Sem esta
// medida eu estaria otimizando por fe.
//
// Compilar:
//   nvcc -O3 -arch=sm_120 codecclean.cu codecclean_bench.cu -o ccbench.exe

#include "codecclean.cuh"
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace cc;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA @%d: %s\n", __LINE__, cudaGetErrorString(e_)); \
    exit(1); } } while (0)

static float cronometra(void (*fn)(void *), void *ctx, int rep)
{
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    fn(ctx); fn(ctx);                       // aquece
    CK(cudaDeviceSynchronize());
    CK(cudaEventRecord(a));
    for (int i = 0; i < rep; ++i) fn(ctx);
    CK(cudaEventRecord(b));
    CK(cudaEventSynchronize(b));
    float ms = 0.0f;
    CK(cudaEventElapsedTime(&ms, a, b));
    CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b));
    return ms / rep;
}

struct Rede
{
    int W, H, ch, bl, cin, modo;
    float *in, *a, *b, *c, *res;
    float *wInp, *bInp, *wOut, *bOut;
    float *w1[8], *b1[8], *w2[8], *b2[8];
    const float *hw;      // peso no HOST, para o modo 3
};

static void roda_rede(void *p)
{
    Rede &r = *(Rede *)p;
    const size_t N = (size_t)r.W * r.H;
    const int m = r.modo;
    conv3_launch(r.in, r.cin, r.a, r.ch, r.wInp, r.bInp, r.W, r.H, 1, 0, 0, m,
                 r.hw, r.hw);
    for (int i = 0; i < r.bl; ++i)
    {
        conv3_launch(r.a, r.ch, r.b, r.ch, r.w1[i], r.b1[i], r.W, r.H, 1, 0, 0,
                     m, r.hw, r.hw);
        if (m == 0)
        {
            // o ingenuo nao sabe somar residuo: precisa do k_add
            conv3_launch(r.b, r.ch, r.c, r.ch, r.w2[i], r.b2[i], r.W, r.H, 0, 0,
                         0, m, r.hw, r.hw);
            k_add<<<(int)((r.ch * N + 255) / 256), 256>>>(r.a, r.c, (int)(r.ch * N));
        }
        else
        {
            conv3_launch(r.b, r.ch, r.a, r.ch, r.w2[i], r.b2[i], r.W, r.H, 0, 1,
                         0, m, r.hw, r.hw);
        }
    }
    conv3_launch(r.a, r.ch, r.res, 1, r.wOut, r.bOut, r.W, r.H, 0, 0, 0, m,
                 r.hw, r.hw);
}

// uma camada 32->32 sozinha, para ter o custo unitario
static void roda_uma(void *p)
{
    Rede &r = *(Rede *)p;
    conv3_launch(r.a, r.ch, r.b, r.ch, r.w1[0], r.b1[0], r.W, r.H, 1, 0, 0,
                 r.modo, r.hw, r.hw);
}

struct Filtro
{
    cc::CodecCleanFilter *f;
    uint8_t *y, *uv, *out;
    int W, H;
};

static void roda_quadro(void *p)
{
    Filtro &f = *(Filtro *)p;
    f.f->push(f.y, f.W, f.uv, f.W);
    const uint8_t *uvo = nullptr; int uvp = 0;
    f.f->pop(f.out, f.W, 1.0f, &uvo, &uvp);
}

int main(int argc, char **argv)
{
    const char *blob = argc > 1 ? argv[1]
                                : "F:/anime-filter-models/engine/cc_32x4.blob";
    const int W = argc > 2 ? atoi(argv[2]) : 1280;
    const int H = argc > 3 ? atoi(argv[3]) : 720;
    const size_t N = (size_t)W * H;

    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s (sm_%d%d, %d SMs)\n", prop.name, prop.major, prop.minor,
           prop.multiProcessorCount);
    printf("quadro: %dx%d\n\n", W, H);

    // ---- 1) quadro inteiro, como o worker paga ----------------------
    cc::CodecCleanFilter filt;
    if (!filt.init(blob, W, H, 0))
    { fprintf(stderr, "init falhou: %s\n", blob); return 1; }

    Filtro fc{&filt, nullptr, nullptr, nullptr, W, H};
    CK(cudaMalloc(&fc.y, N));
    CK(cudaMalloc(&fc.uv, N / 2));
    CK(cudaMalloc(&fc.out, N));
    CK(cudaMemset(fc.y, 120, N));
    CK(cudaMemset(fc.uv, 128, N / 2));
    for (int i = 0; i < 8; ++i) filt.push(fc.y, W, fc.uv, W);   // enche a janela

    float msQuadro = cronometra(roda_quadro, &fc, 100);

    // ---- 2) so a rede, nos dois caminhos ----------------------------
    Rede r{};
    r.W = W; r.H = H; r.ch = 32; r.bl = 4; r.cin = 11; r.modo = 0;
    CK(cudaMalloc(&r.in, r.cin * N * 4));
    CK(cudaMalloc(&r.a, r.ch * N * 4));
    CK(cudaMalloc(&r.b, r.ch * N * 4));
    CK(cudaMalloc(&r.c, r.ch * N * 4));
    CK(cudaMalloc(&r.res, N * 4));
    size_t maiorW = (size_t)r.ch * r.ch * 9;
    std::vector<float> hw(maiorW);
    for (size_t i = 0; i < maiorW; ++i) hw[i] = 0.01f * ((i % 17) - 8);
    r.hw = hw.data();     // o modo 3 le o peso do HOST
    auto sobe = [&](float **d, size_t n) {
        CK(cudaMalloc(d, n * 4));
        CK(cudaMemcpy(*d, hw.data(), n * 4, cudaMemcpyHostToDevice));
    };
    sobe(&r.wInp, (size_t)r.ch * r.cin * 9); sobe(&r.bInp, r.ch);
    sobe(&r.wOut, (size_t)r.ch * 9);         sobe(&r.bOut, 1);
    for (int i = 0; i < r.bl; ++i) {
        sobe(&r.w1[i], maiorW); sobe(&r.b1[i], r.ch);
        sobe(&r.w2[i], maiorW); sobe(&r.b2[i], r.ch);
    }
    CK(cudaMemset(r.in, 0, r.cin * N * 4));
    CK(cudaMemset(r.a, 0, r.ch * N * 4));

    r.modo = 0; float msLenta  = cronometra(roda_rede, &r, 20);
    r.modo = 0; float msUma0   = cronometra(roda_uma, &r, 20);
    r.modo = 3; float msRapida = cronometra(roda_rede, &r, 100);
    r.modo = 3; float msUma1   = cronometra(roda_uma, &r, 100);
    r.modo = 1; float msConst  = cronometra(roda_rede, &r, 100);
    r.modo = 2; float msGlobal = cronometra(roda_rede, &r, 100);

    // ---- relatorio ---------------------------------------------------
    // GMAC da rede: (cin*ch + 2*bl*ch*ch + ch) * 9 por pixel
    double macs = ((double)r.cin * r.ch + 2.0 * r.bl * r.ch * r.ch + r.ch)
                  * 9.0 * (double)N;
    printf("quadro inteiro (push+pop): %8.3f ms   -> %.1f fps\n",
           msQuadro, 1000.0 / msQuadro);
    printf("  a rede dentro dele:      %8.3f ms  (%.1f%%)\n",
           msRapida, 100.0 * msRapida / msQuadro);
    printf("  mapas, io, dither:       %8.3f ms  (%.1f%%)\n",
           msQuadro - msRapida, 100.0 * (msQuadro - msRapida) / msQuadro);

    printf("\n---- a rede, os dois caminhos ----\n");
    printf("  modo 0 ingenuo: %8.3f ms   1,00x  %6.2f TFLOP/s\n",
           msLenta, 2.0 * macs / (msLenta * 1e-3) / 1e12);
    printf("  modo 3 parametro:%7.3f ms  %5.2fx  %6.2f TFLOP/s  <- producao\n",
           msRapida, msLenta / msRapida, 2.0 * macs / (msRapida * 1e-3) / 1e12);
    printf("  modo 1 constante:%7.3f ms  %5.2fx  %6.2f TFLOP/s\n",
           msConst, msLenta / msConst, 2.0 * macs / (msConst * 1e-3) / 1e12);
    printf("  modo 2 global:  %8.3f ms  %5.2fx  %6.2f TFLOP/s\n",
           msGlobal, msLenta / msGlobal, 2.0 * macs / (msGlobal * 1e-3) / 1e12);
    printf("\nATENCAO: este bench roda o filtro SOZINHO. O modo 1 ganha aqui e\n"
           "PERDE no worker (41,92 contra 16,38 ms/quadro do modo 2), porque\n"
           "escrever na memoria de constante drena o dispositivo quando o VSR\n"
           "esta rodando em outro stream. Bench de componente nao prova\n"
           "desempenho de sistema.\n");
    printf("\n  uma conv 32->32: ingenuo %6.3f | rapido %6.3f ms  (%.2fx)\n",
           msUma0, msUma1, msUma0 / msUma1);
    printf("\n  rede: %.2f GMAC/quadro\n", macs / 1e9);

    filt.destroy();
    return 0;
}
