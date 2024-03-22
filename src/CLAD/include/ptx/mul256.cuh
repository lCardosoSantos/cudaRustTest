// Copyright 2022-2024 Dag Arne Osvik
// Copyright 2022-2024 Luan Cardoso dos Santos

#ifndef MUL256_CUH
#define MUL256_CUH

#include <cstdint>

#include "ptx.cuh"

__device__ __forceinline__ void mul256(
    uint64_t &z0,
    uint64_t &z1,
    uint64_t &z2,
    uint64_t &z3,
    uint64_t &z4,
    uint64_t &z5,
    uint64_t &z6,
    uint64_t &z7,
    uint64_t  x0,
    uint64_t  x1,
    uint64_t  x2,
    uint64_t  x3,
    uint64_t  y0,
    uint64_t  y1,
    uint64_t  y2,
    uint64_t  y3)
{
    uint32_t
        xl0, xl1, xl2, xl3,
        xh0, xh1, xh2, xh3,
        yl0, yl1, yl2, yl3,
        yh0, yh1, yh2, yh3,
        l0, l1, h0, h1;

    uint64_t t0, t1, t2, t3, t4, t5, t6, t7;

    unpack(xl0, xh0, x0);
    unpack(xl1, xh1, x1);
    unpack(xl2, xh2, x2);
    unpack(xl3, xh3, x3);

    unpack(yl0, yh0, y0);
    unpack(yl1, yh1, y1);
    unpack(yl2, yh2, y2);
    unpack(yl3, yh3, y3);

    // xl*yl and xh*yh -> z

    mul_wide_u32(z0, xl0, yl0);
    mul_wide_u32(z1, xl0, yl1);
    mul_wide_u32(z2, xl0, yl2);
    mul_wide_u32(z3, xl0, yl3);

    // xl*yh and xh*yl -> t

    mul_wide_u32(t0, xl0, yh0);
    mul_wide_u32(t1, xl0, yh1);
    mul_wide_u32(t2, xl0, yh2);
    mul_wide_u32(t3, xl0, yh3);

    mad_wide_cc_u32 (z1, xh0, yh0, z1);
    madc_wide_cc_u32(z2, xh0, yh1, z2);
    madc_wide_cc_u32(z3, xh0, yh2, z3);
    madc_wide_u32   (z4, xh0, yh3,  0);

    mad_wide_cc_u32 (t0, xh0, yl0, t0);
    madc_wide_cc_u32(t1, xh0, yl1, t1);
    madc_wide_cc_u32(t2, xh0, yl2, t2);
    madc_wide_cc_u32(t3, xh0, yl3, t3);
    addc_u64        (t4, 0, 0);

    mad_wide_cc_u32 (z1, xl1, yl0, z1);
    madc_wide_cc_u32(z2, xl1, yl1, z2);
    madc_wide_cc_u32(z3, xl1, yl2, z3);
    madc_wide_cc_u32(z4, xl1, yl3, z4);
    addc_u64        (z5, 0, 0);

    mad_wide_cc_u32 (t1, xl1, yh0, t1);
    madc_wide_cc_u32(t2, xl1, yh1, t2);
    madc_wide_cc_u32(t3, xl1, yh2, t3);
    madc_wide_u32   (t4, xl1, yh3, t4);

    mad_wide_cc_u32 (z2, xh1, yh0, z2);
    madc_wide_cc_u32(z3, xh1, yh1, z3);
    madc_wide_cc_u32(z4, xh1, yh2, z4);
    madc_wide_cc_u32(z5, xh1, yh3, z5);

    mad_wide_cc_u32 (t1, xh1, yl0, t1);
    madc_wide_cc_u32(t2, xh1, yl1, t2);
    madc_wide_cc_u32(t3, xh1, yl2, t3);
    madc_wide_cc_u32(t4, xh1, yl3, t4);
    addc_u64        (t5, 0, 0);

    mad_wide_cc_u32 (z2, xl2, yl0, z2);
    madc_wide_cc_u32(z3, xl2, yl1, z3);
    madc_wide_cc_u32(z4, xl2, yl2, z4);
    madc_wide_cc_u32(z5, xl2, yl3, z5);
    addc_u64        (z6, 0, 0);

    mad_wide_cc_u32 (t2, xl2, yh0, t2);
    madc_wide_cc_u32(t3, xl2, yh1, t3);
    madc_wide_cc_u32(t4, xl2, yh2, t4);
    madc_wide_u32   (t5, xl2, yh3, t5);

    mad_wide_cc_u32 (z3, xh2, yh0, z3);
    madc_wide_cc_u32(z4, xh2, yh1, z4);
    madc_wide_cc_u32(z5, xh2, yh2, z5);
    madc_wide_cc_u32(z6, xh2, yh3, z6);

    mad_wide_cc_u32 (t2, xh2, yl0, t2);
    madc_wide_cc_u32(t3, xh2, yl1, t3);
    madc_wide_cc_u32(t4, xh2, yl2, t4);
    madc_wide_cc_u32(t5, xh2, yl3, t5);
    addc_u64        (t6, 0, 0);

    mad_wide_cc_u32 (z3, xl3, yl0, z3);
    madc_wide_cc_u32(z4, xl3, yl1, z4);
    madc_wide_cc_u32(z5, xl3, yl2, z5);
    madc_wide_cc_u32(z6, xl3, yl3, z6);
    addc_u64        (z7, 0, 0);

    mad_wide_cc_u32 (t3, xl3, yh0, t3);
    madc_wide_cc_u32(t4, xl3, yh1, t4);
    madc_wide_cc_u32(t5, xl3, yh2, t5);
    madc_wide_u32   (t6, xl3, yh3, t6);

    mad_wide_cc_u32 (z4, xh3, yh0, z4);
    madc_wide_cc_u32(z5, xh3, yh1, z5);
    madc_wide_cc_u32(z6, xh3, yh2, z6);
    madc_wide_cc_u32(z7, xh3, yh3, z7);

    mad_wide_cc_u32 (t3, xh3, yl0, t3);
    madc_wide_cc_u32(t4, xh3, yl1, t4);
    madc_wide_cc_u32(t5, xh3, yl2, t5);
    madc_wide_cc_u32(t6, xh3, yl3, t6);
    addc_u64        (t7, 0, 0);

    // z += t >> 32

    unpack(l0, h0, z0);                           unpack(l1, h1, t0);  add_cc_u32 (h0, h0, l1); pack(z0, l0, h0);
    unpack(l0, h0, z1);  addc_cc_u32(l0, l0, h1); unpack(l1, h1, t1);  addc_cc_u32(h0, h0, l1); pack(z1, l0, h0);
    unpack(l0, h0, z2);  addc_cc_u32(l0, l0, h1); unpack(l1, h1, t2);  addc_cc_u32(h0, h0, l1); pack(z2, l0, h0);
    unpack(l0, h0, z3);  addc_cc_u32(l0, l0, h1); unpack(l1, h1, t3);  addc_cc_u32(h0, h0, l1); pack(z3, l0, h0);
    unpack(l0, h0, z4);  addc_cc_u32(l0, l0, h1); unpack(l1, h1, t4);  addc_cc_u32(h0, h0, l1); pack(z4, l0, h0);
    unpack(l0, h0, z5);  addc_cc_u32(l0, l0, h1); unpack(l1, h1, t5);  addc_cc_u32(h0, h0, l1); pack(z5, l0, h0);
    unpack(l0, h0, z6);  addc_cc_u32(l0, l0, h1); unpack(l1, h1, t6);  addc_cc_u32(h0, h0, l1); pack(z6, l0, h0);
    unpack(l0, h0, z7);  addc_cc_u32(l0, l0, h1); unpack(l1, h1, t7);  addc_cc_u32(h0, h0, l1); pack(z7, l0, h0);
}

#endif
// vim: ts=4 et sw=4 si
