// GATE DA JANELA: a CodecCleanFilter perde quadro, duplica ou troca?
//
// O gate anterior (codecclean_gate.cu) julgou a CONTA — kernels contra
// a referencia de Python. Este julga o FLUXO: push/pop/drain, que e
// onde este worker ja sangrou (iter 92/93: quadro 0 duplicado, ultimo
// perdido, DETERMINISTICO). Contagem errada nao aparece como erro — o
// arquivo abre, roda e tem quase a duracao certa.
//
// Tres perguntas, e nenhuma delas o gate da conta responde:
//   1. entram N quadros, saem N? (nem N-3, nem N+1)
//   2. a ORDEM se mantem? (o quadro que sai e o que devia sair)
//   3. o k=0 e bypass EXATO? (o slider so e seguro se for)
//
// Compilar:
//   nvcc -O2 -arch=sm_120 codecclean.cu codecclean_wingate.cu -o wingate.exe

#include "codecclean.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA @%d: %s\n", __LINE__, cudaGetErrorString(e_)); \
    exit(1); } } while (0)

int main(int argc, char **argv)
{
    const char *blob = argc > 1 ? argv[1] : "cc_32x4.blob";
    const int W = 256, H = 128, NF = 40;   // pequeno: o teste e de FLUXO
    const size_t N = (size_t)W * H;

    cc::CodecCleanFilter filt;
    if (!filt.init(blob, W, H, 0))
    { fprintf(stderr, "init falhou (blob: %s)\n", blob); return 1; }
    printf("janela: %dx%d, %d quadros de teste\n\n", W, H, NF);

    // Cada quadro recebe um valor CONSTANTE = seu indice. Assim o quadro
    // que sai se identifica sozinho: se sair na ordem errada ou repetido,
    // o valor denuncia. Textura nao serve para isso — valor serve.
    uint8_t *dIn, *dOut, *dUV;
    CK(cudaMalloc(&dIn, N));
    CK(cudaMalloc(&dUV, N / 2));
    CK(cudaMemset(dUV, 128, N / 2));
    CK(cudaMalloc(&dOut, N));

    std::vector<int> saiu;
    std::vector<uint8_t> host(N);

    auto drenar = [&](float k) {
        while (filt.available() > 0)
        {
            if (!filt.pop(dOut, W, k, nullptr, nullptr)) break;
            CK(cudaMemcpy(host.data(), dOut, N, cudaMemcpyDeviceToHost));
            // o pixel central diz de que quadro veio (bypass) ou o quanto
            // o filtro mexeu (k=1)
            saiu.push_back((int)host[(H / 2) * W + W / 2]);
        }
    };

    // ---- 1 e 2: contagem e ordem, com k=0 (bypass) ---------------------
    // Com k=0 a saida DEVE ser o proprio quadro de entrada, entao o valor
    // constante atravessa e a ordem fica legivel.
    for (int i = 0; i < NF; ++i)
    {
        CK(cudaMemset(dIn, (int)(i + 1), N));   // +1: evita o valor 0
        filt.push(dIn, W, dUV, W);
        drenar(0.0f);
    }
    filt.markEnd();
    drenar(0.0f);
    CK(cudaDeviceSynchronize());

    printf("1) CONTAGEM: entraram %d, sairam %d  -> %s\n",
           NF, (int)saiu.size(), (int)saiu.size() == NF ? "OK" : "FALHA");

    bool ordem = true;
    for (int i = 0; i < (int)saiu.size(); ++i)
        if (saiu[i] != i + 1) { ordem = false; break; }
    printf("2) ORDEM:    %s", ordem ? "OK" : "FALHA");
    if (!ordem)
    {
        printf("  (esperado 1,2,3...; veio ");
        for (int i = 0; i < 8 && i < (int)saiu.size(); ++i)
            printf("%d ", saiu[i]);
        printf("...)");
    }
    printf("\n");

    // 3) k=0 e bypass EXATO? Ja esta provado acima se a ordem bateu: o
    //    valor saiu intacto. Mas o dither podia mover +-1 e passar
    //    despercebido num pixel so, entao confere o plano inteiro.
    bool bypass = true;
    for (size_t i = 0; i < N; ++i)
        if (host[i] != (uint8_t)NF) { bypass = false; break; }
    printf("3) BYPASS k=0: %s (plano inteiro do ultimo quadro)\n",
           bypass ? "EXATO" : "FALHA — dither ou soma mexeram em k=0");

    bool ok = ((int)saiu.size() == NF) && ordem && bypass;
    printf("\nGATE DA JANELA: %s\n", ok ? "PASSA" : "FALHA");
    filt.destroy();
    return ok ? 0 : 1;
}
