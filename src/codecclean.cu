// CodecClean — filtro de residuo de compressao, em CUDA, dentro do worker.
//
// POR QUE ESTE ARQUIVO EXISTE: o filtro roda hoje em PyTorch. Para
// embarcar num app, o torch custa 4,3 GB (medido) num pacote que inteiro
// tem 291 MB. A rede sao 254k -> 77k parametros: **0,30 MB de pesos**.
// A conta de embarque nao fecha de outro jeito.
//
// O QUE ELE FAZ: recebe uma janela de 7 quadros de LUMA (o plano Y de
// NV12, 8 bits, 0..255), calcula 3 mapas de condicionamento, roda uma
// rede convolucional pequena e devolve um RESIDUO que se soma ao quadro
// central. UV nunca e tocado.
//
// ESPECIFICACAO EXECUTAVEL, e ela nao e opcional:
//   rede   -> eval/gate_engine.py, impl "numpy"   (erro 5e-5 luma)
//   mapas  -> eval/cond_ref.py                    (erro <= 5e-5)
// Os dois ja passam nos gates E1/E2/E3 e P1 do lado de Python. Este
// arquivo tem que reproduzir AQUELES numeros; qualquer divergencia e
// bug deste arquivo, nao da referencia.
//
// TRES DECISOES QUE UM PORT ERRA CALADO, e por isso estao explicitas:
//
//  1. BORDA REFLECT_101 em todo blur: abcd -> cba|abcd|dcb, espelha SEM
//     repetir a borda. E o default do OpenCV, que gerou os mapas do
//     TREINO. BORDER_REPLICATE daria outro numero em 1,4% do quadro
//     (borda de 8 px em 720p) — erro que nao aparece como erro.
//
//  2. A VARIANCIA usa E[g^2] - E[g]^2 com g em 0..255, que e instavel
//     (subtrai numeros da ordem de 65025 para achar algo em 0..12).
//     **NAO CONSERTAR.** O alvo e reproduzir o OpenCV que gerou o
//     treino; uma variancia "melhor" da a rede um canal que ela nunca
//     viu.
//
//  3. A CONVOLUCAO E CROSS-CORRELATION (a do PyTorch), nao convolucao
//     matematica: o kernel NAO e espelhado. Espelhar da uma saida
//     plausivel e errada.

#include "codecclean.cuh"

namespace cc
{

// Kernels gaussianos EXTRAIDOS do cv2 uma vez (resposta ao impulso) e
// congelados. O lado de Python levanta em tempo de execucao porque tem
// OpenCV; aqui nao ha, entao viram tabela. Simetricos, somam 1.
__constant__ float c_g1[9] = {
    0.0001338306f, 0.0044318613f, 0.0539911264f, 0.2419714432f,
    0.3989434769f, 0.2419714432f, 0.0539911264f, 0.0044318613f,
    0.0001338306f};
__constant__ float c_g2[17] = {
    0.0000669163f, 0.0004363490f, 0.0022159630f, 0.0087643040f,
    0.0269959590f, 0.0647599400f, 0.1209874890f, 0.1760357591f,
    0.1994746410f, 0.1760357591f, 0.1209874890f, 0.0647599400f,
    0.0269959590f, 0.0087643040f, 0.0022159630f, 0.0004363490f,
    0.0000669163f};

// Espelhamento _101: o vizinho de -1 e 1, o de n e n-2. O `while` cobre
// o caso patologico de raio maior que a dimensao; em 720p com raio 8
// ele nunca roda mais de uma volta, mas um port em 64x64 sem ele leria
// fora do buffer.
__device__ __forceinline__ int refl101(int i, int n)
{
    if (n == 1)
        return 0;
    while (i < 0 || i >= n)
    {
        if (i < 0)
            i = -i;
        if (i >= n)
            i = 2 * (n - 1) - i;
    }
    return i;
}

__device__ __forceinline__ float clampf(float v, float lo, float hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}

// ---------------------------------------------------------------- blur
// Separavel em duas passadas. Duas passadas e nao uma fundida de
// proposito: o kernel de 17 taps em 2D leria 289 amostras por pixel
// contra 34 assim.

__global__ void k_blur_h(const float *__restrict__ src, float *__restrict__ dst,
                         int w, int h, int ksize)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h)
        return;
    const float *k = (ksize == 9) ? c_g1 : c_g2;
    int r = ksize / 2;
    float a = 0.0f;
    for (int i = 0; i < ksize; ++i)
        a += k[i] * src[y * w + refl101(x + i - r, w)];
    dst[y * w + x] = a;
}

__global__ void k_blur_v(const float *__restrict__ src, float *__restrict__ dst,
                         int w, int h, int ksize)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h)
        return;
    const float *k = (ksize == 9) ? c_g1 : c_g2;
    int r = ksize / 2;
    float a = 0.0f;
    for (int i = 0; i < ksize; ++i)
        a += k[i] * src[refl101(y + i - r, h) * w + x];
    dst[y * w + x] = a;
}

// ------------------------------------------------------------ box 7x7
// boxFilter NORMALIZADO com borda reflect_101, como o cv2.

__global__ void k_box7(const float *__restrict__ src, float *__restrict__ dst,
                       int w, int h)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h)
        return;
    float a = 0.0f;
    for (int dy = -3; dy <= 3; ++dy)
    {
        int yy = refl101(y + dy, h);
        for (int dx = -3; dx <= 3; ++dx)
            a += src[yy * w + refl101(x + dx, w)];
    }
    dst[y * w + x] = a * (1.0f / 49.0f);
}

// --------------------------------------------------------------- mapas

// uint8 -> float 0..255, e o quadrado para a variancia numa passada so.
__global__ void k_u8_to_f32(const uint8_t *__restrict__ src, int pitch,
                            float *__restrict__ dst, float *__restrict__ dst2,
                            int w, int h)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h)
        return;
    float v = (float)src[y * pitch + x];
    dst[y * w + x] = v;
    if (dst2)
        dst2[y * w + x] = v * v;
}

// flat = clip((12 - var) / 12), var = E[g^2] - E[g]^2. Instavel de
// proposito — ver nota 2 no topo.
__global__ void k_flat(const float *__restrict__ mu, const float *__restrict__ mu2,
                       float *__restrict__ dst, int w, int h)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= w * h)
        return;
    float var = mu2[i] - mu[i] * mu[i];
    if (var < 0.0f)
        var = 0.0f;
    dst[i] = clampf((12.0f - var) / 12.0f, 0.0f, 1.0f);
}

// Costura de grade 4/8/16. Cada pixel decide sozinho se cai numa linha
// da grade, e acumula por MAXIMO — por isso a ordem entre os tres `p`
// nao importa e nao ha corrida entre eles.
__global__ void k_seam(const float *__restrict__ g, float *__restrict__ dst,
                       int w, int h)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h)
        return;
    float m = 0.0f;
    for (int p = 4; p <= 16; p *= 2)
    {
        // coluna da grade: dx[x] = |g[x+1] - g[x]|, valido ate w-2
        if (x < w - 1 && ((x + 1) % p) == 0)
        {
            float d = fabsf(g[y * w + x + 1] - g[y * w + x]);
            m = fmaxf(m, clampf((d - 1.0f) * 0.25f, 0.0f, 1.0f));
        }
        if (y < h - 1 && ((y + 1) % p) == 0)
        {
            float d = fabsf(g[(y + 1) * w + x] - g[y * w + x]);
            m = fmaxf(m, clampf((d - 1.0f) * 0.25f, 0.0f, 1.0f));
        }
    }
    dst[y * w + x] = m;
}

// cv2.dilate 3x3 de uns = maximo na janela. A borda do cv2 para dilate
// e -infinito; como a entrada aqui e >= 0, reflect_101 da o mesmo
// numero e evita um caso especial.
__global__ void k_dilate3(const float *__restrict__ src, float *__restrict__ dst,
                          int w, int h)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h)
        return;
    float m = -1e30f;
    for (int dy = -1; dy <= 1; ++dy)
    {
        int yy = refl101(y + dy, h);
        for (int dx = -1; dx <= 1; ++dx)
            m = fmaxf(m, src[yy * w + refl101(x + dx, w)]);
    }
    dst[y * w + x] = m;
}

// grain: |hf(t) - hf(t-1)| acumulado sobre os pares (2,1) (3,2) (4,3),
// onde hf(i) = frame[i] - blur(frame[i], sigma 1).
__global__ void k_grain_acc(const float *__restrict__ f, const float *__restrict__ b,
                            float *__restrict__ acc, int w, int h, int primeiro)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= w * h)
        return;
    float hf = f[i] - b[i];
    if (primeiro)
        acc[i] = hf;          // guarda hf(t-1) na primeira chamada
    else
        acc[i] = fabsf(hf - acc[i]);
}

__global__ void k_clamp01_scale(float *__restrict__ x, float s, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        x[i] = clampf(x[i] * s, 0.0f, 1.0f);
}

// Escala SEM clampar. A referencia faz `clip(blur(acc/3), 0, 1)`: a
// divisao vem ANTES do blur e o clamp DEPOIS. Clampar antes corta picos
// que o blur ainda ia espalhar pelos vizinhos — erro que fica invisivel
// na mediana (0,00002) e aparece so no maximo (0,70 luma), porque so
// atinge os poucos pixels onde acc/3 passa de 1.
__global__ void k_scale(float *__restrict__ x, float s, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        x[i] = x[i] * s;
}

// --------------------------------------------------------------- rede

// Convolucao 3x3 generica, padding ZERO (o do nn.Conv2d(padding=1)) —
// NAO reflect. Os mapas usam reflect porque o OpenCV usa; a rede usa
// zero porque o PyTorch usa. Misturar os dois e o erro mais facil de
// cometer neste arquivo.
//
// CROSS-CORRELATION: o kernel nao e espelhado (nota 3 no topo).
// Peso em layout do PyTorch: [out][in][ky][kx].
__global__ void k_conv3(const float *__restrict__ src, int cin,
                        float *__restrict__ dst, int cout,
                        const float *__restrict__ wt, const float *__restrict__ bias,
                        int w, int h, int relu)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int o = blockIdx.z;
    if (x >= w || y >= h || o >= cout)
        return;

    float a = bias[o];
    for (int c = 0; c < cin; ++c)
    {
        const float *plane = src + (size_t)c * w * h;
        const float *k = wt + ((size_t)o * cin + c) * 9;
        for (int ky = 0; ky < 3; ++ky)
        {
            int yy = y + ky - 1;
            if (yy < 0 || yy >= h)
                continue;              // padding ZERO
            for (int kx = 0; kx < 3; ++kx)
            {
                int xx = x + kx - 1;
                if (xx < 0 || xx >= w)
                    continue;
                a += k[ky * 3 + kx] * plane[yy * w + xx];
            }
        }
    }
    if (relu && a < 0.0f)
        a = 0.0f;
    dst[(size_t)o * w * h + y * w + x] = a;
}

// x = x + r  (a soma do ResBlock)
__global__ void k_add(float *__restrict__ x, const float *__restrict__ r, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        x[i] += r[i];
}

// ------------------------------------------------------------- saida
//
// out = clamp(deg + k * clamp(residuo, -1, 1) * 255, 0, 255), com
// DITHER antes do floor.
//
// O dither nao e enfeite: o residuo do modelo e ASSIMETRICO (ele
// escurece), e arredondar direto deixa vies de brilho que MUDA com a
// forca — medido -0,157 luma em k=0,30 no lado de Python. Num slider
// isso seria fatal: mexer na forca moveria a exposicao junto.
//
// A soma e em DOUBLE pelo mesmo motivo do lado de Python: em float o
// espacamento perto de 235 e ~1,5e-5, entao 235 + 0,99999994 sobe um
// nivel mesmo com dither correto — e quebra a promessa de BYPASS EXATO
// em k=0, que e o que faz o slider ser seguro.
__global__ void k_compose(const uint8_t *__restrict__ deg, int pitchIn,
                          const float *__restrict__ res,
                          const float *__restrict__ ruido,
                          uint8_t *__restrict__ out, int pitchOut,
                          float k, int w, int h)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h)
        return;
    float r = clampf(res[y * w + x], -1.0f, 1.0f) * 255.0f * k;
    double v = (double)deg[y * pitchIn + x] + (double)r;
    if (v < 0.0)
        v = 0.0;
    if (v > 255.0)
        v = 255.0;
    double q = floor(v + (double)ruido[y * w + x]);
    if (q < 0.0)
        q = 0.0;
    if (q > 255.0)
        q = 255.0;
    out[y * pitchOut + x] = (uint8_t)q;
}

} // namespace cc
