// Copyright 2022-2024 Dag Arne Osvik
// Copyright 2022-2024 Luan Cardoso dos Santos

#ifndef FP_REDUCE5_CUH
#define FP_REDUCE5_CUH

#include <cstdint>

#include "ptx.cuh"

__device__ __forceinline__ void fp_reduce5(
    uint64_t &z0, uint64_t &z1, uint64_t &z2, uint64_t &z3,
    uint64_t  x0, uint64_t  x1, uint64_t  x2, uint64_t  x3, uint64_t  x4
    )
{
    assert(x4 < 12);    // This function is only intended for reduction of values up to 260 bits

    uint32_t
        t0, t1, t2, t3, t4, t5, t6, t7, t8, t9,
        l0, l1, l2, l3, l4, l5, l6, l7,
        h0, h1, h2, h3, h4, h5, h6, h7;

    uint64_t
        u0, u1, u2, u3, u4, u5, u6, u7;

    unpack(t0, t1, x0);
    unpack(t2, t3, x1);
    unpack(t4, t5, x2);
    unpack(t6, t7, x3);
    unpack(t8, t9, x4); // the value in t9 is not used 

    shf_l_wrap_b32(t9, t7, t8, 4);  // q1

    mul_lo_u32(t9, t9, 0x54);       // q2

    shr_b32(t9, t9, 8);             // q3

    assert(t9 < 0x80);

    // q3 * m

    mul_wide_u32(u0, t9, 0xD87CFD47);
    mul_wide_u32(u1, t9, 0x3C208C16);
    mul_wide_u32(u2, t9, 0x6871CA8D);
    mul_wide_u32(u3, t9, 0x97816A91);
    mul_wide_u32(u4, t9, 0x8181585D);
    mul_wide_u32(u5, t9, 0xB85045B6);
    mul_wide_u32(u6, t9, 0xE131A029);
    mul_wide_u32(u7, t9, 0x30644E72);

    unpack(l0, h0, u0);
    unpack(l1, h1, u1);
    unpack(l2, h2, u2);
    unpack(l3, h3, u3);
    unpack(l4, h4, u4);
    unpack(l5, h5, u5);
    unpack(l6, h6, u6);
    unpack(l7, h7, u7);

    add_cc_u32 (h0, h0, l1);
    addc_cc_u32(l2, l2, h1);
    addc_cc_u32(h2, h2, l3);
    addc_cc_u32(l4, l4, h3);
    addc_cc_u32(h4, h4, l5);
    addc_cc_u32(l6, l6, h5);
    addc_u32   (h6, h6, l7);

    pack(u0, l0, h0);
    pack(u1, l2, h2);
    pack(u2, l4, h4);
    pack(u3, l6, h6);

    // r = r1 - r2 = x - q3 mod 2^256

    sub_cc_u64 (z0, x0, u0);
    subc_cc_u64(z1, x1, u1);
    subc_cc_u64(z2, x2, u2);
    subc_u64   (z3, x3, u3);
}

#endif

// vim: ts=4 et sw=4 si
