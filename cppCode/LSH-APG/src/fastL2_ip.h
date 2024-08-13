#pragma once

#define USE_FAST



#ifdef USE_FAST

#define __SSE__
#define __AVX__

#ifdef __SSE__
#define USE_SSE
#ifdef __AVX__
#define USE_AVX
#endif
#endif
#endif

#if defined(USE_AVX) || defined(USE_SSE)
#ifdef _MSC_VER
#include <intrin.h>
#include <stdexcept>
#else
#include <x86intrin.h>
#endif

#if defined(__GNUC__)
#define PORTABLE_ALIGN32 __attribute__((aligned(32)))
#else
#define PORTABLE_ALIGN32 __declspec(align(32))
#endif
#endif

namespace fastlib {

    static float
        L2Sqr(float* pVect1, float* pVect2, size_t qty) {
        float res = 0;
        for (size_t i = 0; i < qty; i++) {
            float t = *pVect1 - *pVect2;
            pVect1++;
            pVect2++;
            res += t * t;
        }
        return (res);
    }

    static float
        InnerProduct(float* pVect1, float* pVect2, size_t qty) {
        float res = 0;
        for (size_t i = 0; i < qty; i++) {
            res += *pVect1 * (*pVect2);
            pVect1++;
            pVect2++;
        }
        return (res);
    }

#if defined(USE_AVX)

    // Favor using AVX if available.
    static float
        L2SqrSIMD16Ext(float* pVect1, float* pVect2, size_t qty) {
        float PORTABLE_ALIGN32 TmpRes[8];
        size_t qty16 = qty >> 4;

        const float* pEnd1 = pVect1 + (qty16 << 4);

        __m256 diff, v1, v2;
        __m256 sum = _mm256_set1_ps(0);

        while (pVect1 < pEnd1) {
            v1 = _mm256_loadu_ps(pVect1);
            pVect1 += 8;
            v2 = _mm256_loadu_ps(pVect2);
            pVect2 += 8;
            diff = _mm256_sub_ps(v1, v2);
            sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));

            v1 = _mm256_loadu_ps(pVect1);
            pVect1 += 8;
            v2 = _mm256_loadu_ps(pVect2);
            pVect2 += 8;
            diff = _mm256_sub_ps(v1, v2);
            sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
        }

        _mm256_store_ps(TmpRes, sum);
        return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] + TmpRes[7];
    }

    static float
        IpSIMD16Ext(float* pVect1, float* pVect2, size_t qty) {
        float PORTABLE_ALIGN32 TmpRes[8];
        size_t qty16 = qty >> 4;

        const float* pEnd1 = pVect1 + (qty16 << 4);

        __m256 v1, v2;
        __m256 sum = _mm256_set1_ps(0);

        while (pVect1 < pEnd1) {
            v1 = _mm256_loadu_ps(pVect1);
            pVect1 += 8;
            v2 = _mm256_loadu_ps(pVect2);
            pVect2 += 8;
            //diff = _mm256_sub_ps(v1, v2);
            sum = _mm256_add_ps(sum, _mm256_mul_ps(v1, v2));

            v1 = _mm256_loadu_ps(pVect1);
            pVect1 += 8;
            v2 = _mm256_loadu_ps(pVect2);
            pVect2 += 8;
            //diff = _mm256_sub_ps(v1, v2);
            sum = _mm256_add_ps(sum, _mm256_mul_ps(v1, v2));
        }

        _mm256_store_ps(TmpRes, sum);
        return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] + TmpRes[7];
    }
#endif

#if defined(USE_SSE) || defined(USE_AVX)
    static float
        L2SqrSIMD16ExtResiduals(float* pVect1v, float* pVect2v, size_t qty) {
        //size_t qty = *((size_t*)qty_ptr);
        size_t qty16 = qty >> 4 << 4;
        float res = L2SqrSIMD16Ext(pVect1v, pVect2v, qty16);
        float* pVect1 = (float*)pVect1v + qty16;
        float* pVect2 = (float*)pVect2v + qty16;

        size_t qty_left = qty - qty16;
        float res_tail = L2Sqr(pVect1, pVect2, qty_left);
        return (res + res_tail);
    }

    static float
        IpSIMD16ExtResiduals(float* pVect1v, float* pVect2v, size_t qty) {
        //size_t qty = *((size_t*)qty_ptr);
        size_t qty16 = qty >> 4 << 4;
        float res = IpSIMD16Ext(pVect1v, pVect2v, qty16);
        float* pVect1 = (float*)pVect1v + qty16;
        float* pVect2 = (float*)pVect2v + qty16;

        size_t qty_left = qty - qty16;
        float res_tail = InnerProduct(pVect1, pVect2, qty_left);
        return (res + res_tail);
    }
#endif


#ifdef USE_SSE
    static float
        L2SqrSIMD4Ext(float* pVect1, float* pVect2, size_t qty) {
        float PORTABLE_ALIGN32 TmpRes[8];

        size_t qty4 = qty >> 2;

        const float* pEnd1 = pVect1 + (qty4 << 2);

        __m128 diff, v1, v2;
        __m128 sum = _mm_set1_ps(0);

        while (pVect1 < pEnd1) {
            v1 = _mm_loadu_ps(pVect1);
            pVect1 += 4;
            v2 = _mm_loadu_ps(pVect2);
            pVect2 += 4;
            diff = _mm_sub_ps(v1, v2);
            sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
        }
        _mm_store_ps(TmpRes, sum);
        return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];
    }

    static float
        IpSIMD4Ext(float* pVect1, float* pVect2, size_t qty) {
        float PORTABLE_ALIGN32 TmpRes[8];
        //float* pVect1 = (float*)pVect1v;
        //float* pVect2 = (float*)pVect2v;
        //size_t qty = *((size_t*)qty_ptr);


        size_t qty4 = qty >> 2;

        const float* pEnd1 = pVect1 + (qty4 << 2);

        __m128 v1, v2;
        __m128 sum = _mm_set1_ps(0);

        while (pVect1 < pEnd1) {
            v1 = _mm_loadu_ps(pVect1);
            pVect1 += 4;
            v2 = _mm_loadu_ps(pVect2);
            pVect2 += 4;
            //diff = _mm_sub_ps(v1, v2);
            sum = _mm_add_ps(sum, _mm_mul_ps(v1, v2));
        }
        _mm_store_ps(TmpRes, sum);
        return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];
    }

    static float
        L2SqrSIMD4ExtResiduals(float* pVect1v, float* pVect2v, size_t qty) {
        //size_t qty = *((size_t*)qty_ptr);
        size_t qty4 = qty >> 2 << 2;

        float res = L2SqrSIMD4Ext(pVect1v, pVect2v, qty4);
        size_t qty_left = qty - qty4;

        float* pVect1 = (float*)pVect1v + qty4;
        float* pVect2 = (float*)pVect2v + qty4;
        float res_tail = L2Sqr(pVect1, pVect2, qty_left);

        return (res + res_tail);
    }

    static float
        IpSIMD4ExtResiduals(float* pVect1v, float* pVect2v, size_t qty) {
        //size_t qty = *((size_t*)qty_ptr);
        size_t qty4 = qty >> 2 << 2;

        float res = IpSIMD4Ext(pVect1v, pVect2v, qty4);
        size_t qty_left = qty - qty4;

        float* pVect1 = (float*)pVect1v + qty4;
        float* pVect2 = (float*)pVect2v + qty4;
        float res_tail = InnerProduct(pVect1, pVect2, qty_left);

        return (res + res_tail);
    }
#endif

}

inline float calL2Sqr_fast(float* v1, float* v2, int dim)
{
#if defined(USE_FAST)
    if (dim % 16 == 0)
        return fastlib::L2SqrSIMD16Ext(v1, v2, dim);
    else if (dim % 4 == 0)
        return fastlib::L2SqrSIMD4Ext(v1, v2, dim);
    else if (dim > 16)
        return fastlib::L2SqrSIMD16ExtResiduals(v1, v2, dim);
    else if (dim > 4)
        return fastlib::L2SqrSIMD4ExtResiduals(v1, v2, dim);
#else
    float res = 0.0;
    for (int i = 0; i < dim; ++i) {
        res += (v1[i] - v2[i]) * (v1[i] - v2[i]);
    }
    return res;
#endif


}

inline float calIp_fast(float* v1, float* v2, int dim)
{
#if defined(USE_FAST)
    if (dim % 16 == 0)
        return fastlib::IpSIMD16Ext(v1, v2, dim);
    else if (dim % 4 == 0)
        return fastlib::IpSIMD4Ext(v1, v2, dim);
    else if (dim > 16)
        return fastlib::IpSIMD16ExtResiduals(v1, v2, dim);
    else if (dim > 4)
        return fastlib::IpSIMD4ExtResiduals(v1, v2, dim);
#else
    float res = 0.0;
    for (int i = 0; i < dim; ++i) {
        res += v1[i] * v2[i];
    }
    return res;
#endif


}

namespace fastlib1 {
    static float
        L2Sqr(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
        float* pVect1 = (float*)pVect1v;
        float* pVect2 = (float*)pVect2v;
        size_t qty = *((size_t*)qty_ptr);

        float res = 0;
        for (size_t i = 0; i < qty; i++) {
            float t = *pVect1 - *pVect2;
            pVect1++;
            pVect2++;
            res += t * t;
        }
        return (res);
    }

#if defined(USE_AVX)

    // Favor using AVX if available.
    static float
        L2SqrSIMD16Ext(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
        float* pVect1 = (float*)pVect1v;
        float* pVect2 = (float*)pVect2v;
        size_t qty = *((size_t*)qty_ptr);
        float PORTABLE_ALIGN32 TmpRes[8];
        size_t qty16 = qty >> 4;

        const float* pEnd1 = pVect1 + (qty16 << 4);

        __m256 diff, v1, v2;
        __m256 sum = _mm256_set1_ps(0);

        while (pVect1 < pEnd1) {
            v1 = _mm256_loadu_ps(pVect1);
            pVect1 += 8;
            v2 = _mm256_loadu_ps(pVect2);
            pVect2 += 8;
            diff = _mm256_sub_ps(v1, v2);
            sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));

            v1 = _mm256_loadu_ps(pVect1);
            pVect1 += 8;
            v2 = _mm256_loadu_ps(pVect2);
            pVect2 += 8;
            diff = _mm256_sub_ps(v1, v2);
            sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
        }

        _mm256_store_ps(TmpRes, sum);
        return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] + TmpRes[7];
    }

#elif defined(USE_SSE)

    static float
        L2SqrSIMD16Ext(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
        float* pVect1 = (float*)pVect1v;
        float* pVect2 = (float*)pVect2v;
        size_t qty = *((size_t*)qty_ptr);
        float PORTABLE_ALIGN32 TmpRes[8];
        size_t qty16 = qty >> 4;

        const float* pEnd1 = pVect1 + (qty16 << 4);

        __m128 diff, v1, v2;
        __m128 sum = _mm_set1_ps(0);

        while (pVect1 < pEnd1) {
            //_mm_prefetch((char*)(pVect2 + 16), _MM_HINT_T0);
            v1 = _mm_loadu_ps(pVect1);
            pVect1 += 4;
            v2 = _mm_loadu_ps(pVect2);
            pVect2 += 4;
            diff = _mm_sub_ps(v1, v2);
            sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

            v1 = _mm_loadu_ps(pVect1);
            pVect1 += 4;
            v2 = _mm_loadu_ps(pVect2);
            pVect2 += 4;
            diff = _mm_sub_ps(v1, v2);
            sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

            v1 = _mm_loadu_ps(pVect1);
            pVect1 += 4;
            v2 = _mm_loadu_ps(pVect2);
            pVect2 += 4;
            diff = _mm_sub_ps(v1, v2);
            sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

            v1 = _mm_loadu_ps(pVect1);
            pVect1 += 4;
            v2 = _mm_loadu_ps(pVect2);
            pVect2 += 4;
            diff = _mm_sub_ps(v1, v2);
            sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
        }

        _mm_store_ps(TmpRes, sum);
        return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];
    }
#endif

#if defined(USE_SSE) || defined(USE_AVX)
    static float
        L2SqrSIMD16ExtResiduals(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
        size_t qty = *((size_t*)qty_ptr);
        size_t qty16 = qty >> 4 << 4;
        float res = L2SqrSIMD16Ext(pVect1v, pVect2v, &qty16);
        float* pVect1 = (float*)pVect1v + qty16;
        float* pVect2 = (float*)pVect2v + qty16;

        size_t qty_left = qty - qty16;
        float res_tail = L2Sqr(pVect1, pVect2, &qty_left);
        return (res + res_tail);
    }
#endif


#ifdef USE_SSE
    static float
        L2SqrSIMD4Ext(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
        float PORTABLE_ALIGN32 TmpRes[8];
        float* pVect1 = (float*)pVect1v;
        float* pVect2 = (float*)pVect2v;
        size_t qty = *((size_t*)qty_ptr);


        size_t qty4 = qty >> 2;

        const float* pEnd1 = pVect1 + (qty4 << 2);

        __m128 diff, v1, v2;
        __m128 sum = _mm_set1_ps(0);

        while (pVect1 < pEnd1) {
            v1 = _mm_loadu_ps(pVect1);
            pVect1 += 4;
            v2 = _mm_loadu_ps(pVect2);
            pVect2 += 4;
            diff = _mm_sub_ps(v1, v2);
            sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
        }
        _mm_store_ps(TmpRes, sum);
        return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];
    }

    static float
        L2SqrSIMD4ExtResiduals(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
        size_t qty = *((size_t*)qty_ptr);
        size_t qty4 = qty >> 2 << 2;

        float res = L2SqrSIMD4Ext(pVect1v, pVect2v, &qty4);
        size_t qty_left = qty - qty4;

        float* pVect1 = (float*)pVect1v + qty4;
        float* pVect2 = (float*)pVect2v + qty4;
        float res_tail = L2Sqr(pVect1, pVect2, &qty_left);

        return (res + res_tail);
    }
#endif
}

inline float calL2Sqr_fast1(float* v1, float* v2, size_t dim)
{
#if defined(USE_FAST)
    if (dim % 16 == 0)
        return fastlib1::L2SqrSIMD16Ext(v1, v2, &dim);
    else if (dim % 4 == 0)
        return fastlib1::L2SqrSIMD4Ext(v1, v2, &dim);
    else if (dim > 16)
        return fastlib1::L2SqrSIMD16ExtResiduals(v1, v2, &dim);
    else if (dim > 4)
        return fastlib1::L2SqrSIMD4ExtResiduals(v1, v2, &dim);
#else
    float res = 0.0;
    for (int i = 0; i < dim; ++i) {
        res += (v1[i] - v2[i]) * (v1[i] - v2[i]);
    }
    return res;
#endif


}