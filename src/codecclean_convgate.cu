// GATE DA CONVOLUCAO: o kernel rapido da o MESMO numero que o ingenuo?
//
// POR QUE ESTE GATE E SEPARADO DO E1: o E1 julga a rede inteira contra
// o golden de Python, com barra de 0,5 luma. E a verdade final, mas e
// GROSSA — um erro de borda em 1% do quadro passa por baixo dela. Aqui
// a barra e outra: os dois kernels somam na MESMA ordem (canal de
// entrada, depois os 9 taps), entao o resultado tem que bater BIT A
// BIT. Qualquer diferenca e sinal de que a ordem mudou, de que a borda
// esta diferente, ou de que o peso foi lido do lugar errado.
//
// Nao e purismo: os tres bugs da integracao so apareceram porque havia
// um teste EXATO no caminho. Os testes com tolerancia deixaram passar.
//
// QUATRO FORMAS, cada uma cobrindo um risco:
//   11->32  a entrada, com cin que NAO e multiplo dos 4 canais por carga
//   32->32  o miolo, 8 das 10 camadas
//   32->1   a saida, um canal so
//   48->48  o controle da fornada, que NAO cabe na memoria de constante
//           e tem que desviar para o caminho global em vez de calar
// E o resid: a soma do ResBlock fundida na conv contra o k_add separado.
//
// Compilar:
//   nvcc -O3 -arch=sm_120 codecclean.cu codecclean_convgate.cu -o convgate.exe

#include "codecclean.cuh"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

using namespace cc;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA @%d: %s\n", __LINE__, cudaGetErrorString(e_)); \
    exit(1); } } while (0)

// gerador proprio: o teste tem que dar o MESMO caso toda vez
static unsigned semente = 20260819u;
static float sorteia()
{
    semente = semente * 1664525u + 1013904223u;
    return ((float)(semente >> 8) / 8388608.0f) - 1.0f;   // -1 .. 1
}

struct Caso { int cin, cout; const char *nome; };

static size_t compara(const std::vector<float> &ref,
                      const std::vector<float> &got, double *pior)
{
    size_t difs = 0;
    *pior = 0.0;
    for (size_t i = 0; i < ref.size(); ++i)
        if (ref[i] != got[i])
        {
            ++difs;
            double d = fabs((double)ref[i] - (double)got[i]);
            if (d > *pior) *pior = d;
        }
    return difs;
}

int main(int argc, char **argv)
{
    // 1277x715 DE PROPOSITO: com tamanho redondo os blocos da borda
    // ficam cheios e o guarda de fronteira nunca e exercitado — o bug
    // de borda so aparece quando o ultimo bloco fica pela metade.
    const int W = argc > 1 ? atoi(argv[1]) : 1277;
    const int H = argc > 2 ? atoi(argv[2]) : 715;
    const size_t N = (size_t)W * H;

    Caso casos[] = {
        {11, 32, "entrada  11->32"},
        {32, 32, "miolo    32->32"},
        {32,  1, "saida    32->1 "},
        {48, 48, "controle 48->48 (fora da constante)"},
    };

    printf("quadro %dx%d (NAO redondo de proposito: exercita a borda)\n\n", W, H);

    bool tudoOk = true;
    for (const Caso &k : casos)
    {
        std::vector<float> hSrc((size_t)k.cin * N), hW((size_t)k.cout * k.cin * 9),
                           hB(k.cout), hPre((size_t)k.cout * N);
        for (float &v : hSrc) v = sorteia();
        for (float &v : hW)   v = sorteia() * 0.1f;
        for (float &v : hB)   v = sorteia() * 0.01f;
        for (float &v : hPre) v = sorteia();          // o que ja esta em dst

        float *dSrc, *dW, *dB, *dRef, *dGot, *dPre;
        CK(cudaMalloc(&dSrc, hSrc.size() * 4));
        CK(cudaMalloc(&dW, hW.size() * 4));
        CK(cudaMalloc(&dB, hB.size() * 4));
        CK(cudaMalloc(&dPre, hPre.size() * 4));
        CK(cudaMalloc(&dRef, (size_t)k.cout * N * 4));
        CK(cudaMalloc(&dGot, (size_t)k.cout * N * 4));
        CK(cudaMemcpy(dSrc, hSrc.data(), hSrc.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dW, hW.data(), hW.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dB, hB.data(), hB.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dPre, hPre.data(), hPre.size() * 4, cudaMemcpyHostToDevice));

        std::vector<float> ref((size_t)k.cout * N), got(ref.size());
        double pior;

        // ---- A) sem residuo, relu ligado ----------------------------
        conv3_launch(dSrc, k.cin, dRef, k.cout, dW, dB, W, H, 1, 0, 0, 0);
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(ref.data(), dRef, ref.size() * 4, cudaMemcpyDeviceToHost));

        CK(cudaMemset(dGot, 0, ref.size() * 4));
        // MODO 3 = o de producao (peso por parametro de kernel). As
        // formas que ele nao cobre caem sozinhas no modo 2 e o gate
        // julga o que de fato rodou.
        bool caiu = !conv3_launch(dSrc, k.cin, dGot, k.cout, dW, dB,
                                  W, H, 1, 0, 0, 3, hW.data(), hB.data());
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(got.data(), dGot, got.size() * 4, cudaMemcpyDeviceToHost));
        size_t difA = compara(ref, got, &pior);
        printf("  %-36s reto:  %s  difs %zu  pior %.3e%s\n", k.nome,
               difA == 0 ? "IDENTICO" : "DIFERE  ", difA, pior,
               caiu ? "  (caiu no ingenuo)" : "");
        tudoOk &= (difA == 0);

        // ---- B) com residuo, sem relu (o caso do ResBlock) ----------
        // referencia: ingenuo escreve em dRef, k_add soma o dPre nele.
        conv3_launch(dSrc, k.cin, dRef, k.cout, dW, dB, W, H, 0, 0, 0, 0);
        k_add<<<(int)((ref.size() + 255) / 256), 256>>>(dRef, dPre,
                                                       (int)ref.size());
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(ref.data(), dRef, ref.size() * 4, cudaMemcpyDeviceToHost));

        // rapido: parte de dPre e soma dentro do proprio kernel
        CK(cudaMemcpy(dGot, dPre, ref.size() * 4, cudaMemcpyDeviceToDevice));
        conv3_launch(dSrc, k.cin, dGot, k.cout, dW, dB, W, H, 0, 1, 0, 3,
                     hW.data(), hB.data());
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(got.data(), dGot, got.size() * 4, cudaMemcpyDeviceToHost));
        size_t difB = compara(ref, got, &pior);
        printf("  %-36s resid: %s  difs %zu  pior %.3e\n", k.nome,
               difB == 0 ? "IDENTICO" : "DIFERE  ", difB, pior);
        tudoOk &= (difB == 0);

        cudaFree(dSrc); cudaFree(dW); cudaFree(dB);
        cudaFree(dPre); cudaFree(dRef); cudaFree(dGot);
    }

    printf("\nGATE DA CONVOLUCAO: %s\n", tudoOk ? "PASSA" : "FALHA");
    return tudoOk ? 0 : 1;
}
