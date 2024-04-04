// Copyright 2022-2024 Dag Arne Osvik
// Copyright 2022-2024 Luan Cardoso dos Santos

#ifndef FP_REDUCE4_CUH
#define FP_REDUCE4_CUH

#include <cstdint>

#include "ptx.cuh"

//Full modular reduction
__device__ __forceinline__ void fp_reduce4(
    uint64_t &z0, uint64_t &z1, uint64_t &z2, uint64_t &z3,
    uint64_t  x0, uint64_t  x1, uint64_t  x2, uint64_t  x3
    )
{
    uint32_t
        q1, q2, q3,
        t0, t1, t2, t3, t4, t5, t6, t7,
        l0, l1, l2, l3, l4, l5, l6, l7,
        h0, h1, h2, h3, h4, h5, h6, h7;

    uint64_t
        u0, u1, u2, u3, u4, u5, u6, u7;

    unpack(t0, t1, x0);
    unpack(t2, t3, x1);
    unpack(t4, t5, x2);
    unpack(t6, t7, x3);

    q1 = t7;

    mul_hi_u32(q2, q1, 0x54A47462); // floor( floor(x/2^224)*floor(2^284/p) / 2^32)

    shr_b32(q3, q2, 28);            // floor( floor(x/2^224)*floor(2^284/p) / 2^60)

    // q3 * m

    mul_wide_u32(u0, q3, 0xD87CFD47);
    mul_wide_u32(u1, q3, 0x3C208C16);
    mul_wide_u32(u2, q3, 0x6871CA8D);
    mul_wide_u32(u3, q3, 0x97816A91);
    mul_wide_u32(u4, q3, 0x8181585D);
    mul_wide_u32(u5, q3, 0xB85045B6);
    mul_wide_u32(u6, q3, 0xE131A029);
    mul_wide_u32(u7, q3, 0x30644E72);

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

    // r = r1 - r2 = x - q3 * m mod 2^256

    sub_cc_u64 (z0, x0, u0);
    subc_cc_u64(z1, x1, u1);
    subc_cc_u64(z2, x2, u2);
    subc_u64   (z3, x3, u3);

    sub_cc_u64 (u0, z0, 0x3C208C16D87CFD47);
    subc_cc_u64(u1, z1, 0x97816A916871CA8D);
    subc_cc_u64(u2, z2, 0xB85045B68181585D);
    subc_u64   (u3, z3, 0x30644E72E131A029);

    if (u3 < z3) // no underflow, so z >= p
    {
        z0 = u0;
        z1 = u1;
        z2 = u2;
        z3 = u3;
    }
}

#endif

// vim: ts=4 et sw=4 si
