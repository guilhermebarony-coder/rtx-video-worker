// Prototipos dos kernels de codecclean.cu.
//
// Header e nao declaracao solta no .cu de teste: sem o `__global__` os
// protótipos viram funcoes de HOST e o linker procura simbolos que nao
// existem (11 LNK2019). E declarar duas vezes e como as duas copias
// divergem depois.
#pragma once
#include <cuda_runtime.h>
#include <stdint.h>

namespace cc
{
__global__ void k_blur_h(const float *src, float *dst, int w, int h, int ksize);
__global__ void k_blur_v(const float *src, float *dst, int w, int h, int ksize);
__global__ void k_box7(const float *src, float *dst, int w, int h);
__global__ void k_u8_to_f32(const uint8_t *src, int pitch, float *dst,
                            float *dst2, int w, int h);
__global__ void k_flat(const float *mu, const float *mu2, float *dst, int w, int h);
__global__ void k_seam(const float *g, float *dst, int w, int h);
__global__ void k_dilate3(const float *src, float *dst, int w, int h);
__global__ void k_grain_acc(const float *f, const float *b, float *acc,
                            int w, int h, int primeiro);
__global__ void k_clamp01_scale(float *x, float s, int n);
// escala SEM clampar: o grain divide por 3 e so clampa DEPOIS do
// blur. Clampar antes corta picos que o blur ainda ia espalhar.
__global__ void k_scale(float *x, float s, int n);
__global__ void k_fill(float *x, float v, int n);
__global__ void k_conv3(const float *src, int cin, float *dst, int cout,
                        const float *wt, const float *bias, int w, int h, int relu);
__global__ void k_add(float *x, const float *r, int n);
__global__ void k_compose(const uint8_t *deg, int pitchIn, const float *res,
                          const float *ruido, uint8_t *out, int pitchOut,
                          float k, int w, int h);

// ---------------------------------------------------------------------
// CodecCleanFilter — o filtro inteiro, com a janela de 7 quadros em VRAM.
//
// CONTRATO: `push` entrega um quadro; `pop` devolve o proximo quadro
// FILTRADO, que fica 3 atras do ultimo entregue. No fim, `flush` drena
// os 3 presos. Quem chama tem que drenar, senao o video termina 3
// quadros mais cedo — e isso NAO aparece como erro (o arquivo abre,
// roda, e tem quase a duracao certa). E o mesmo modo de falha do
// iter 92/93 deste worker.
//
// A janela e enderecada por `absoluto % 7`: ao andar para i+1 entra
// i+4, cujo slot e o mesmo de i-3, exatamente o quadro que saiu. O 7
// nao e folga, e a identidade que fecha o esquema sem caso especial —
// e indices repetidos no clamp das pontas apontam para o mesmo slot,
// tambem sem caso especial.
class CodecCleanFilter
{
public:
    // blobPath: pesos exportados por eval/dump_blob.py
    bool init(const char *blobPath, int w, int h, cudaStream_t stream);
    void destroy();

    // Entrega um quadro. O CROMA entra junto e viaja na mesma janela:
    // o filtro atrasa o luma em 3 quadros, e se o croma nao for
    // atrasado igual a cor sai 3 quadros a frente da imagem.
    void push(const uint8_t *d_y, int pitchY,
              const uint8_t *d_uv, int pitchUV);

    // Quantos quadros filtrados ja podem sair.
    int available() const;

    // Escreve o proximo quadro filtrado. `strength` e o k do slider:
    // 0 = bypass EXATO, 1 = dose cheia. Devolve false se nao ha nada.
    // `d_uvOut` recebe o ponteiro do croma CORRESPONDENTE ao quadro
    // que saiu (dentro do ring, valido ate 4 pushes depois).
    bool pop(uint8_t *d_out, int outPitch, float strength,
             const uint8_t **d_uvOut, int *uvPitchOut);

    // Sinaliza fim da entrada: os ultimos quadros passam a poder sair
    // com a vizinhanca CLAMPADA (repete o ultimo), como nas pontas do
    // lado de Python.
    void markEnd() { m_ended = true; }

    int width() const { return m_w; }
    int height() const { return m_h; }

private:
    void runNetwork(int centro, float strength, uint8_t *d_out, int outPitch);

    int m_w = 0, m_h = 0, m_ch = 0, m_bl = 0, m_cin = 0;
    long long m_pushed = 0;   // quantos quadros entraram
    long long m_popped = 0;   // quantos sairam
    bool m_ended = false;
    cudaStream_t m_stream = nullptr;

    uint8_t *m_ring = nullptr;      // 7 planos Y contiguos, pitch = m_w
    uint8_t *m_ringUV = nullptr;    // 7 planos UV, pitch = m_w, altura h/2
    float *m_w8 = nullptr;          // pesos
    float *m_in = nullptr, *m_a = nullptr, *m_b = nullptr, *m_c = nullptr;
    float *m_f = nullptr, *m_f2 = nullptr, *m_t1 = nullptr, *m_t2 = nullptr;
    float *m_acc = nullptr, *m_soma = nullptr, *m_res = nullptr, *m_noise = nullptr;
    float *m_wInp = nullptr, *m_bInp = nullptr, *m_wOut = nullptr, *m_bOut = nullptr;
    float *m_w1[8] = {}, *m_b1[8] = {}, *m_w2[8] = {}, *m_b2[8] = {};
};

// Ruido de dither DETERMINISTICO por (quadro, x, y) — sem estado e sem
// gerador. NAO reproduz a sequencia do numpy/torch do lado de Python, e
// isso e declarado: dither e ruido de media zero por definicao, e o que
// o gate cobra e o VIES de brilho, nao a igualdade do ruido.
__global__ void k_dither(float *out, int w, int h, unsigned frameIdx);

} // namespace cc
