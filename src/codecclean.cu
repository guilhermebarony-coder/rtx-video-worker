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
#include <cstdio>
#include <cstring>
#include <cstdlib>

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

__global__ void k_fill(float *__restrict__ x, float v, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        x[i] = v;
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

// ------------------------------------------------- convolucao rapida
//
// Pesos da camada CORRENTE. Memoria de constante: 64 KB no total, e o
// acesso uniforme (todo thread do warp le o mesmo endereco) e servido
// pelo banco de constantes direto como operando da instrucao.
//
// CUIDADO: isto e do MODULO, nao da instancia. Duas CodecCleanFilter
// concorrentes se sobrescreveriam. O init recusa a segunda.
__constant__ float c_cw[9216];   // ate 32x32x9
__constant__ float c_cb[64];

// Tile de 32x8 pixels de saida com halo de 1 -> 34x10 na shared.
#define CC_TW 34
#define CC_TH 10

// COUT em template para os acumuladores ficarem em REGISTRADOR: com
// indice de tempo de execucao o array vai para memoria local e o kernel
// fica mais lento que o ingenuo.
// ------------------------------------------- convolucao rapida, v2
//
// CG canais de entrada por carga: as leituras vao juntas para a memoria
// (uma cobre a latencia da outra) e o numero de barreiras cai por CG.
#define CC_CG 4

// CIN_T = 0 -> cin de tempo de execucao. Diferente de zero, o indice do
// peso vira deslocamento imediato e some a multiplicacao inteira.
template <int COUT, int CIN_T, int CONSTW>
__global__ void k_conv3_tile2(const float *__restrict__ src, int cinRT,
                              float *__restrict__ dst,
                              const float *__restrict__ wt,
                              const float *__restrict__ bias,
                              int w, int h, int relu, int resid)
{
    __shared__ float sm[CC_CG][CC_TH][CC_TW];

    const int cin = CIN_T ? CIN_T : cinRT;
    const int bx = blockIdx.x * 32, by = blockIdx.y * 8;
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int x = bx + tx, y = by + ty;
    const int tid = ty * 32 + tx;
    const size_t N = (size_t)w * h;

    float acc[COUT];
#pragma unroll
    for (int o = 0; o < COUT; ++o)
        acc[o] = CONSTW ? c_cb[o] : bias[o];

    for (int c0 = 0; c0 < cin; c0 += CC_CG)
    {
        __syncthreads();
#pragma unroll
        for (int g = 0; g < CC_CG; ++g)
        {
            if (c0 + g >= cin)
                break;
            const float *plane = src + (size_t)(c0 + g) * N;
            for (int t = tid; t < CC_TW * CC_TH; t += 256)
            {
                int sy = t / CC_TW, sx = t - sy * CC_TW;
                int gy = by + sy - 1, gx = bx + sx - 1;
                // padding ZERO, o do nn.Conv2d(padding=1)
                sm[g][sy][sx] = (gx >= 0 && gx < w && gy >= 0 && gy < h)
                                    ? plane[(size_t)gy * w + gx]
                                    : 0.0f;
            }
        }
        __syncthreads();

#pragma unroll
        for (int g = 0; g < CC_CG; ++g)
        {
            if (c0 + g >= cin)
                break;
            float v[9];
#pragma unroll
            for (int ky = 0; ky < 3; ++ky)
#pragma unroll
                for (int kx = 0; kx < 3; ++kx)
                    v[ky * 3 + kx] = sm[g][ty + ky][tx + kx];

            const int base = (c0 + g) * 9;
#pragma unroll
            for (int o = 0; o < COUT; ++o)
            {
                const int wo = o * cin * 9 + base;
#pragma unroll
                for (int t = 0; t < 9; ++t)
                    acc[o] = fmaf(CONSTW ? c_cw[wo + t] : wt[wo + t],
                                  v[t], acc[o]);
            }
        }
    }

    if (x >= w || y >= h)
        return;
    // `resid` = a saida SOMA no que ja esta em dst (a ponte do ResBlock).
    // A soma vem DEPOIS da acumulacao inteira, igual ao k_add que ela
    // substitui — mesma ordem, mesmo numero.
    if (resid)
    {
#pragma unroll
        for (int o = 0; o < COUT; ++o)
        {
            size_t idx = (size_t)o * N + (size_t)y * w + x;
            float a = acc[o];
            if (relu && a < 0.0f)
                a = 0.0f;
            dst[idx] += a;
        }
    }
    else
    {
#pragma unroll
        for (int o = 0; o < COUT; ++o)
        {
            float a = acc[o];
            if (relu && a < 0.0f)
                a = 0.0f;
            dst[(size_t)o * N + (size_t)y * w + x] = a;
        }
    }
}

// ---------------------------------------- conv com peso em PARAMETRO
//
// Parametro de kernel mora no banco de constantes 0 e e escrito pela
// via normal do lancamento. Da o mesmo operando imediato da FFMA que a
// memoria de constante dava, SEM `cudaMemcpyToSymbol` — que dentro do
// worker custava mais caro que o kernel ingenuo inteiro (medido:
// 41,92 contra 37,45 ms/quadro, com o ingenuo perdendo so para si
// mesmo no bench).
//
// Teto de 32.764 bytes por lancamento: por isso 16 canais de saida por
// vez (16x32x9x4 + 64 = 18.496 B). Dois lancamentos por camada.
template <int COUT, int CIN_T>
struct PesosKP
{
    float w[COUT * CIN_T * 9];
    float b[COUT];
};

template <int COUT, int CIN_T>
__global__ void k_conv3_kp(__grid_constant__ const PesosKP<COUT, CIN_T> pk,
                           const float *__restrict__ src,
                           float *__restrict__ dst, int oBase,
                           int w, int h, int relu, int resid)
{
    __shared__ float sm[CC_CG][CC_TH][CC_TW];

    const int cin = CIN_T;
    const int bx = blockIdx.x * 32, by = blockIdx.y * 8;
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int x = bx + tx, y = by + ty;
    const int tid = ty * 32 + tx;
    const size_t N = (size_t)w * h;

    float acc[COUT];
#pragma unroll
    for (int o = 0; o < COUT; ++o)
        acc[o] = pk.b[o];

    for (int c0 = 0; c0 < cin; c0 += CC_CG)
    {
        __syncthreads();
#pragma unroll
        for (int g = 0; g < CC_CG; ++g)
        {
            if (c0 + g >= cin)
                break;
            const float *plane = src + (size_t)(c0 + g) * N;
            for (int t = tid; t < CC_TW * CC_TH; t += 256)
            {
                int sy = t / CC_TW, sx = t - sy * CC_TW;
                int gy = by + sy - 1, gx = bx + sx - 1;
                sm[g][sy][sx] = (gx >= 0 && gx < w && gy >= 0 && gy < h)
                                    ? plane[(size_t)gy * w + gx]
                                    : 0.0f;      // padding ZERO
            }
        }
        __syncthreads();

#pragma unroll
        for (int g = 0; g < CC_CG; ++g)
        {
            if (c0 + g >= cin)
                break;
            float v[9];
#pragma unroll
            for (int ky = 0; ky < 3; ++ky)
#pragma unroll
                for (int kx = 0; kx < 3; ++kx)
                    v[ky * 3 + kx] = sm[g][ty + ky][tx + kx];

            const int base = (c0 + g) * 9;
#pragma unroll
            for (int o = 0; o < COUT; ++o)
            {
                const int wo = o * CIN_T * 9 + base;
#pragma unroll
                for (int t = 0; t < 9; ++t)
                    acc[o] = fmaf(pk.w[wo + t], v[t], acc[o]);
            }
        }
    }

    if (x >= w || y >= h)
        return;
    // MESMA ordem de soma do caminho ingenuo: o residuo entra depois da
    // acumulacao inteira. O gate cobra igualdade BIT A BIT.
    if (resid)
    {
#pragma unroll
        for (int o = 0; o < COUT; ++o)
        {
            float a = acc[o];
            if (relu && a < 0.0f)
                a = 0.0f;
            dst[(size_t)(oBase + o) * N + (size_t)y * w + x] += a;
        }
    }
    else
    {
#pragma unroll
        for (int o = 0; o < COUT; ++o)
        {
            float a = acc[o];
            if (relu && a < 0.0f)
                a = 0.0f;
            dst[(size_t)(oBase + o) * N + (size_t)y * w + x] = a;
        }
    }
}

// Monta o grupo de COUT canais e lanca. `hw`/`hb` sao os pesos DESTA
// camada no HOST: o parametro do kernel e copiado pelo driver a cada
// lancamento, entao eles precisam estar do lado de ca.
template <int COUT, int CIN_T>
static void lanca_kp(const float *hw, const float *hb, int oBase,
                     const float *src, float *dst, int w, int h,
                     int relu, int resid, cudaStream_t st)
{
    PesosKP<COUT, CIN_T> pk;
    memcpy(pk.w, hw + (size_t)oBase * CIN_T * 9,
           sizeof(float) * COUT * CIN_T * 9);
    memcpy(pk.b, hb + oBase, sizeof(float) * COUT);
    dim3 blk(32, 8), grd((w + 31) / 32, (h + 7) / 8);
    k_conv3_kp<COUT, CIN_T><<<grd, blk, 0, st>>>(pk, src, dst, oBase,
                                                 w, h, relu, resid);
}

// Launcher: escolhe o caminho e some com o template do lado de fora.
// modo 0 = ingenuo (referencia), 1 = constante, 2 = global.
// Devolve false se caiu no ingenuo por falta de especializacao.
bool conv3_launch(const float *src, int cin, float *dst, int cout,
                  const float *wt, const float *bias, int w, int h,
                  int relu, int resid, cudaStream_t st, int modo,
                  const float *hostW, const float *hostB)
{
    dim3 blk(32, 8), grd((w + 31) / 32, (h + 7) / 8);
    const size_t nw = (size_t)cout * cin * 9;

    // MODO 3: peso como parametro do kernel. So as formas do campeao —
    // o teto de 32.764 bytes por lancamento nao deixa generalizar, e
    // quem nao cabe cai no modo 2, que e correto e so mais lento.
    if (modo == 3)
    {
        if (hostW && hostB && cout == 32 && cin == 32)
        {
            lanca_kp<16, 32>(hostW, hostB, 0, src, dst, w, h, relu, resid, st);
            lanca_kp<16, 32>(hostW, hostB, 16, src, dst, w, h, relu, resid, st);
            return true;
        }
        if (hostW && hostB && cout == 32 && cin == 11)
        {
            lanca_kp<16, 11>(hostW, hostB, 0, src, dst, w, h, relu, resid, st);
            lanca_kp<16, 11>(hostW, hostB, 16, src, dst, w, h, relu, resid, st);
            return true;
        }
        if (hostW && hostB && cout == 1 && cin == 32)
        {
            lanca_kp<1, 32>(hostW, hostB, 0, src, dst, w, h, relu, resid, st);
            return true;
        }
        modo = 2;    // forma nao coberta (ou sem peso no host): global
    }

    // A constante tem 64 KB. Cabe 32x32x9 (36 KB); NAO cabe 48x48x9
    // (81 KB), que e o modelo de controle da fornada. Quando nao cabe,
    // o peso vem da global — mais lento, mas roda.
    // modo 2 = mesmo tile, peso na global: para quem NAO e dono da
    // memoria de constante, e para pesos que nao cabem nela.
    const int constw = (modo == 1) && (nw <= 9216 && cout <= 64);
    if (constw)
    {
        cudaMemcpyToSymbolAsync(c_cw, wt, nw * sizeof(float), 0,
                                cudaMemcpyDeviceToDevice, st);
        cudaMemcpyToSymbolAsync(c_cb, bias, (size_t)cout * sizeof(float), 0,
                                cudaMemcpyDeviceToDevice, st);
    }

#define CC_D2(CO, CI)                                                          \
    do {                                                                       \
        if (constw) k_conv3_tile2<CO, CI, 1><<<grd, blk, 0, st>>>(              \
                        src, cin, dst, wt, bias, w, h, relu, resid);           \
        else        k_conv3_tile2<CO, CI, 0><<<grd, blk, 0, st>>>(              \
                        src, cin, dst, wt, bias, w, h, relu, resid);           \
        return true;                                                           \
    } while (0)

    if (modo != 0)
    {
        // formas do campeao com cin fixo em template: o indice do peso
        // vira deslocamento imediato e some a multiplicacao inteira
        if (cout == 32 && cin == 32) CC_D2(32, 32);
        if (cout == 32 && cin == 11) CC_D2(32, 11);
        if (cout == 1  && cin == 32) CC_D2(1, 32);
        switch (cout)
        {
        case 1:  CC_D2(1, 0);
        case 16: CC_D2(16, 0);
        case 24: CC_D2(24, 0);
        case 32: CC_D2(32, 0);
        case 48: CC_D2(48, 0);
        default: break;    // largura nao especializada -> ingenuo
        }
    }
#undef CC_D2

    // Caminho ingenuo: a REFERENCIA. Nao sabe somar residuo, e nao
    // precisa — quem chama com resid=1 esta no caminho rapido.
    dim3 gz((w + 31) / 32, (h + 7) / 8, cout);
    k_conv3<<<gz, blk, 0, st>>>(src, cin, dst, cout, wt, bias, w, h, relu);
    return modo == 0;
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



// ---------------------------------------------------------------------
// Ruido de dither por hash de (quadro, x, y). Determinista, sem estado.
// Um gerador com estado exigiria que a ordem dos quadros fosse fixa, e
// aqui ela ja e — mas hash tambem sobrevive a reprocessar um quadro
// isolado, que e util para depurar.
__global__ void k_dither(float *__restrict__ out, int w, int h, unsigned frameIdx)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h)
        return;
    unsigned s = frameIdx * 2654435761u ^ (unsigned)x * 2246822519u
               ^ (unsigned)y * 3266489917u;
    s ^= s >> 15; s *= 2246822519u;
    s ^= s >> 13; s *= 3266489917u;
    s ^= s >> 16;
    out[y * w + x] = (float)(s >> 8) * (1.0f / 16777216.0f);  // [0,1)
}

// ---------------------------------------------------------------------

#define CCK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) return false; } while (0)

// Quem manda na memoria de constante. Ver a nota no topo de c_cw: ela
// e do modulo, entao so uma instancia pode escrever nela.
static CodecCleanFilter *s_donoDaConstante = nullptr;

bool CodecCleanFilter::init(const char *blobPath, int w, int h, cudaStream_t stream)
{
    m_w = w; m_h = h; m_stream = stream;
    m_pushed = m_popped = 0; m_ended = false;

    // A primeira instancia fica dona da constante; as demais caem no
    // caminho global. Sem isto, duas instancias em streams diferentes
    // se sobrescreveriam e o defeito sairia em pixel, nao em erro.
    // Modo 3 (peso por parametro) e o de producao: da o operando
    // imediato da FFMA sem escrever em memoria de constante. O dono da
    // constante so importa se alguem FORCAR o modo 1 pelo CC_MODO.
    m_modoConv = 3;
    if (s_donoDaConstante == nullptr)
        s_donoDaConstante = this;

    // Escotilha de diagnostico: CC_MODO=0/1/2 forca o caminho da
    // convolucao. Existe porque medida sintetica e medida no relogio
    // discordaram por 5x e so o A/B dentro do worker desempata.
    if (const char *e = getenv("CC_MODO"))
    {
        int m = atoi(e);
        if (m == 1 && s_donoDaConstante != this)
        {
            fprintf(stderr, "[codecclean] CC_MODO=1 pedido mas a constante "
                            "ja tem dono; usando 2\n");
            m = 2;
        }
        if (m >= 0 && m <= 3)
        {
            m_modoConv = m;
            fprintf(stderr, "[codecclean] CC_MODO=%d forcado%s\n", m,
                    m == 0 ? " (kernel ingenuo, LENTO de proposito)" : "");
        }
    }

    FILE *f = fopen(blobPath, "rb");
    if (!f)
        return false;
    char magia[8];
    int hdr[4];
    if (fread(magia, 1, 8, f) != 8 || memcmp(magia, "CCNET\0\0\0", 8) != 0
        || fread(hdr, 4, 4, f) != 4)
    { fclose(f); return false; }
    m_ch = hdr[0]; m_bl = hdr[1]; m_cin = 7 + hdr[2];
    if (m_bl > 8) { fclose(f); return false; }   // m_w1[] tem 8 slots

    size_t nW = (size_t)m_ch * m_cin * 9 + m_ch;
    for (int b = 0; b < m_bl; ++b) nW += 2 * ((size_t)m_ch * m_ch * 9 + m_ch);
    nW += (size_t)m_ch * 9 + 1;
    float *hostW = (float *)malloc(nW * 4);
    if (!hostW || fread(hostW, 4, nW, f) != nW)
    { free(hostW); fclose(f); return false; }
    fclose(f);

    const size_t N = (size_t)m_w * m_h;
    bool ok = true;
    ok &= cudaMalloc(&m_w8, nW * 4) == cudaSuccess;
    ok &= cudaMalloc(&m_ring, 7 * N) == cudaSuccess;
    // NV12: o plano de croma tem metade da altura e a mesma largura
    ok &= cudaMalloc(&m_ringUV, 7 * (size_t)m_w * (m_h / 2)) == cudaSuccess;
    ok &= cudaMalloc(&m_in, (size_t)m_cin * N * 4) == cudaSuccess;
    ok &= cudaMalloc(&m_a, (size_t)m_ch * N * 4) == cudaSuccess;
    ok &= cudaMalloc(&m_b, (size_t)m_ch * N * 4) == cudaSuccess;
    for (float **p : {&m_f, &m_f2, &m_t1, &m_t2, &m_acc, &m_soma, &m_res, &m_noise})
        ok &= cudaMalloc(p, N * 4) == cudaSuccess;
    if (!ok) { free(hostW); destroy(); return false; }

    cudaMemcpy(m_w8, hostW, nW * 4, cudaMemcpyHostToDevice);
    // O peso FICA no host tambem: o modo 3 passa por parametro de
    // kernel, e o driver copia do lado de ca a cada lancamento. Sao
    // 310 KB para o campeao — barato perto de drenar o dispositivo 10
    // vezes por quadro, que era o custo da memoria de constante.
    m_hostW = hostW;

    // offsets na MESMA ordem do dump_blob — contrato, nao convencao
    size_t o = 0;
    m_wInp = m_w8 + o; o += (size_t)m_ch * m_cin * 9;
    m_bInp = m_w8 + o; o += m_ch;
    for (int b = 0; b < m_bl; ++b) {
        m_w1[b] = m_w8 + o; o += (size_t)m_ch * m_ch * 9;
        m_b1[b] = m_w8 + o; o += m_ch;
        m_w2[b] = m_w8 + o; o += (size_t)m_ch * m_ch * 9;
        m_b2[b] = m_w8 + o; o += m_ch;
    }
    m_wOut = m_w8 + o; o += (size_t)m_ch * 9;
    m_bOut = m_w8 + o;
    return true;
}

void CodecCleanFilter::destroy()
{
    if (s_donoDaConstante == this)
        s_donoDaConstante = nullptr;
    if (m_hostW) { free(m_hostW); m_hostW = nullptr; }
    for (void *p : {(void *)m_w8, (void *)m_ring, (void *)m_ringUV,
                    (void *)m_in, (void *)m_a,
                    (void *)m_b, (void *)m_f, (void *)m_f2,
                    (void *)m_t1, (void *)m_t2, (void *)m_acc, (void *)m_soma,
                    (void *)m_res, (void *)m_noise})
        if (p) cudaFree(p);
    m_w8 = nullptr; m_ring = nullptr; m_ringUV = nullptr;
    m_in = m_a = m_b = nullptr;
    m_f = m_f2 = m_t1 = m_t2 = m_acc = m_soma = m_res = m_noise = nullptr;
}

void CodecCleanFilter::push(const uint8_t *d_y, int pitchY,
                            const uint8_t *d_uv, int pitchUV)
{
    const size_t N = (size_t)m_w * m_h;
    const size_t NUV = (size_t)m_w * (m_h / 2);
    int slot = (int)(m_pushed % 7);
    cudaMemcpy2DAsync(m_ring + (size_t)slot * N, m_w, d_y, pitchY,
                      m_w, m_h, cudaMemcpyDeviceToDevice, m_stream);
    // O croma NAO passa pelo filtro (o modelo so mexe no luma), mas
    // TEM que viajar junto: sem isso a cor sai 3 quadros a frente da
    // imagem, e nem contagem nem PTS denunciam.
    if (d_uv)
        cudaMemcpy2DAsync(m_ringUV + (size_t)slot * NUV, m_w, d_uv, pitchUV,
                          m_w, m_h / 2, cudaMemcpyDeviceToDevice, m_stream);
    ++m_pushed;
}

int CodecCleanFilter::available() const
{
    // O quadro `c` so pode sair quando c+3 ja entrou. No fim da entrada
    // a vizinhanca futura passa a ser CLAMPADA e todos podem sair.
    long long pronto = m_ended ? m_pushed : (m_pushed - 3);
    long long n = pronto - m_popped;
    return n > 0 ? (int)n : 0;
}

bool CodecCleanFilter::pop(uint8_t *d_out, int outPitch, float strength,
                           const uint8_t **d_uvOut, int *uvPitchOut)
{
    if (available() <= 0)
        return false;
    const int centro = (int)m_popped;
    runNetwork(centro, strength, d_out, outPitch);
    if (d_uvOut)
    {
        const size_t NUV = (size_t)m_w * (m_h / 2);
        *d_uvOut = m_ringUV + (size_t)(centro % 7) * NUV;
        if (uvPitchOut)
            *uvPitchOut = m_w;
    }
    ++m_popped;
    return true;
}

// O miolo: monta a entrada de 11 canais e roda a rede sobre o quadro
// `centro` (indice ABSOLUTO). A vizinhanca e clampada em [0, ultimo],
// igual ao lado de Python — nas pontas os indices repetem e caem no
// mesmo slot do ring, sem caso especial.
void CodecCleanFilter::runNetwork(int centro, float strength,
                                  uint8_t *d_out, int outPitch)
{
    const int W = m_w, H = m_h;
    const size_t N = (size_t)W * H;
    dim3 blk(32, 8), grd((W + 31) / 32, (H + 7) / 8);
    int lin = (int)((N + 255) / 256);
    cudaStream_t st = m_stream;

    // ultimo quadro valido: durante o stream e o que ja entrou; depois
    // do markEnd e o ultimo de todos.
    long long ultimo = m_pushed - 1;

    auto slotDe = [&](long long abs) -> const uint8_t * {
        if (abs < 0) abs = 0;
        if (abs > ultimo) abs = ultimo;
        return m_ring + (size_t)(abs % 7) * N;
    };

    // --- canais 0..6: os 7 quadros normalizados 0..1 -------------------
    for (int k = 0; k < 7; ++k)
    {
        const uint8_t *src = slotDe((long long)centro + k - 3);
        k_u8_to_f32<<<grd, blk, 0, st>>>(src, W, m_in + (size_t)k * N,
                                         nullptr, W, H);
        k_scale<<<lin, 256, 0, st>>>(m_in + (size_t)k * N, 1.0f / 255.0f, (int)N);
    }

    // --- o quadro CENTRAL em 0..255, e o quadrado, para os mapas -------
    const uint8_t *dCentro = slotDe(centro);
    k_u8_to_f32<<<grd, blk, 0, st>>>(dCentro, W, m_f, m_f2, W, H);

    // canal 7: bloco
    k_seam<<<grd, blk, 0, st>>>(m_f, m_t1, W, H);
    k_dilate3<<<grd, blk, 0, st>>>(m_t1, m_t2, W, H);
    k_blur_h<<<grd, blk, 0, st>>>(m_t2, m_t1, W, H, 17);
    k_blur_v<<<grd, blk, 0, st>>>(m_t1, m_in + (size_t)7 * N, W, H, 17);
    k_clamp01_scale<<<lin, 256, 0, st>>>(m_in + (size_t)7 * N, 1.0f, (int)N);

    // canal 8: flat
    k_box7<<<grd, blk, 0, st>>>(m_f, m_t1, W, H);
    k_box7<<<grd, blk, 0, st>>>(m_f2, m_t2, W, H);
    k_flat<<<lin, 256, 0, st>>>(m_t1, m_t2, m_in + (size_t)8 * N, W, H);

    // canal 9: grao — |hf(i) - hf(i-1)| para i em {2,3,4} da janela.
    // A divisao por 3 vem ANTES do blur e o clamp DEPOIS: clampar antes
    // corta picos que o blur ainda ia espalhar (bug pego pelo gate, que
    // aparecia so no maximo e sumia na mediana).
    cudaMemsetAsync(m_soma, 0, N * 4, st);
    for (int i = 2; i <= 4; ++i)
    {
        for (int par = 0; par < 2; ++par)
        {
            const uint8_t *src = slotDe((long long)centro + (i - par) - 3);
            k_u8_to_f32<<<grd, blk, 0, st>>>(src, W, m_f, nullptr, W, H);
            k_blur_h<<<grd, blk, 0, st>>>(m_f, m_t1, W, H, 9);
            k_blur_v<<<grd, blk, 0, st>>>(m_t1, m_t2, W, H, 9);
            k_grain_acc<<<lin, 256, 0, st>>>(m_f, m_t2, m_acc, W, H, par == 0);
        }
        k_add<<<lin, 256, 0, st>>>(m_soma, m_acc, (int)N);
    }
    k_scale<<<lin, 256, 0, st>>>(m_soma, 1.0f / 3.0f, (int)N);
    k_blur_h<<<grd, blk, 0, st>>>(m_soma, m_t1, W, H, 17);
    k_blur_v<<<grd, blk, 0, st>>>(m_t1, m_in + (size_t)9 * N, W, H, 17);
    k_clamp01_scale<<<lin, 256, 0, st>>>(m_in + (size_t)9 * N, 1.0f, (int)N);

    // canal 10: GOP. O lado de Python usa o pict_type do stream original
    // (I=0, P=1, B=2) dividido por 2. Aqui o worker ainda nao propaga
    // isso ate o filtro, entao vai o valor de P — o mais comum. ISTO E
    // UMA DIVERGENCIA DECLARADA, nao um esquecimento: o gate do port
    // roda com o mesmo valor dos dois lados, entao ela nao se esconde
    // ali; e o dia que o pict_type chegar aqui, este e o unico ponto a
    // mudar.
    {
        k_fill<<<lin, 256, 0, st>>>(m_in + (size_t)10 * N, 0.5f, (int)N);
    }

    // --- a rede --------------------------------------------------------
    // MODO 3: peso como parametro do kernel. Nao e o mais rapido no
    // bench por muito — mas e o unico que nao escreve em memoria de
    // constante, e essa escrita custa 28 ms/quadro DENTRO do worker
    // (41,92 do modo 1 contra 13,72 deste). conv3_launch cai sozinho
    // para o modo 2 nas formas que nao cabem em 32 KB de parametro.
    const int modo = m_modoConv;
    // O host guarda o MESMO leiaute do device: o deslocamento de cada
    // camada e a diferenca dos ponteiros de device. Sem membro novo.
    const float *HW = m_hostW;
    auto hp = [&](const float *dev) { return HW ? HW + (dev - m_w8) : nullptr; };
    conv3_launch(m_in, m_cin, m_a, m_ch, m_wInp, m_bInp, W, H, 1, 0, st, modo,
                 hp(m_wInp), hp(m_bInp));
    for (int b = 0; b < m_bl; ++b)
    {
        conv3_launch(m_a, m_ch, m_b, m_ch, m_w1[b], m_b1[b], W, H, 1, 0, st,
                     modo, hp(m_w1[b]), hp(m_b1[b]));
        // resid=1: a segunda conv soma direto em m_a. Sem isto haveria
        // um k_add movendo 354 MB por bloco, 1,4 GB por quadro.
        conv3_launch(m_b, m_ch, m_a, m_ch, m_w2[b], m_b2[b], W, H, 0, 1, st,
                     modo, hp(m_w2[b]), hp(m_b2[b]));
    }
    conv3_launch(m_a, m_ch, m_res, 1, m_wOut, m_bOut, W, H, 0, 0, st, modo,
                 hp(m_wOut), hp(m_bOut));

    // --- composicao: deg + k*residuo, com dither e soma em double ------
    k_dither<<<grd, blk, 0, st>>>(m_noise, W, H, (unsigned)centro);
    k_compose<<<grd, blk, 0, st>>>(dCentro, W, m_res, m_noise,
                                   d_out, outPitch, strength, W, H);
}

} // namespace cc
