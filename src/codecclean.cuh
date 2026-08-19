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
__global__ void k_conv3(const float *src, int cin, float *dst, int cout,
                        const float *wt, const float *bias, int w, int h, int relu);
__global__ void k_add(float *x, const float *r, int n);
__global__ void k_compose(const uint8_t *deg, int pitchIn, const float *res,
                          const float *ruido, uint8_t *out, int pitchOut,
                          float k, int w, int h);
}
