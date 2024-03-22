// Finite Field Arithmetic for BN254
// Copyright 2022-2024 Dag Arne Osvik
// Copyright 2022-2024 Luan Cardoso dos Santos

#ifndef FR_REDUCE8_CUH
#define FR_REDUCE8_CUH

#include <cassert>

#include "fr.cuh"
#include "ptx.cuh"

__device__ __forceinline__ void fr_reduce8(
    uint64_t &z0, uint64_t &z1, uint64_t &z2, uint64_t &z3,
    uint64_t  x0, uint64_t  x1, uint64_t  x2, uint64_t  x3,
    uint64_t  x4, uint64_t  x5, uint64_t  x6, uint64_t  x7
    )
{
    const uint32_t 

        // r

        m7 = 0x30644E72,
        m6 = 0xE131A029,
        m5 = 0xB85045B6,
        m4 = 0x8181585D,
        m3 = 0x2833E848,
        m2 = 0x79B97091,
        m1 = 0x43E1F593,
        m0 = 0xF0000001,

    // 2^512/r

        mu8 = 0x00000005,
        mu7 = 0x4A474626,
        mu6 = 0x23A04A7A,
        mu5 = 0xB074A586,
        mu4 = 0x80730147,
        mu3 = 0x14485200,
        mu2 = 0x9E880AE6,
        mu1 = 0x20703A6B,
        mu0 = 0xE1DE9259;

    uint32_t
        t0, t1, t2, t3, t4, t5, t6, t7,
        t8, t9, ta, tb, tc, td, te, tf,
        l0, l1, h0, h1;

    uint64_t
        q0, q1, q2, q3, q4,
        r0, r1, r2, r3, r4,
        u0, u1, u2, u3, u4;

    unpack(t0, t1, x0);
    unpack(t2, t3, x1);
    unpack(t4, t5, x2);
    unpack(t6, t7, x3);
    unpack(t8, t9, x4);
    unpack(ta, tb, x5);
    unpack(tc, td, x6);
    unpack(te, tf, x7);

    mul_wide_u32(u0, mu8, t7);
    mul_wide_u32(q0, mu8, t8);


    mad_wide_cc_u32 (u0, mu7, t8, u0);
    addc_u64        (u1, 0, 0);       

    mad_wide_cc_u32 (q0, mu7, t9, q0);
    addc_u64        (q1, 0, 0);       


    mad_wide_cc_u32(u0, mu6, t9, u0); 
    madc_wide_u32  (u1, mu8, t9, u1); 

    mad_wide_cc_u32(q0, mu6, ta, q0); 
    madc_wide_u32  (q1, mu8, ta, q1); 


    mad_wide_cc_u32 (u0, mu5, ta, u0);
    madc_wide_cc_u32(u1, mu7, ta, u1);
    addc_u64        (u2, 0, 0);       

    mad_wide_cc_u32 (q0, mu5, tb, q0);
    madc_wide_cc_u32(q1, mu7, tb, q1);
    addc_u64        (q2, 0, 0);       


    mad_wide_cc_u32 (u0, mu4, tb, u0);
    madc_wide_cc_u32(u1, mu6, tb, u1);
    madc_wide_u32   (u2, mu8, tb, u2);

    mad_wide_cc_u32 (q0, mu4, tc, q0);
    madc_wide_cc_u32(q1, mu6, tc, q1);
    madc_wide_u32   (q2, mu8, tc, q2);


    mad_wide_cc_u32 (u0, mu3, tc, u0);
    madc_wide_cc_u32(u1, mu5, tc, u1);
    madc_wide_cc_u32(u2, mu7, tc, u2);
    addc_u64        (u3, 0, 0);       

    mad_wide_cc_u32 (q0, mu3, td, q0);
    madc_wide_cc_u32(q1, mu5, td, q1);
    madc_wide_cc_u32(q2, mu7, td, q2);
    addc_u64        (q3, 0, 0);       


    mad_wide_cc_u32 (u0, mu2, td, u0);
    madc_wide_cc_u32(u1, mu4, td, u1);
    madc_wide_cc_u32(u2, mu6, td, u2);
    madc_wide_u32   (u3, mu8, td, u3);

    mad_wide_cc_u32 (q0, mu2, te, q0);
    madc_wide_cc_u32(q1, mu4, te, q1);
    madc_wide_cc_u32(q2, mu6, te, q2);
    madc_wide_u32   (q3, mu8, te, q3);


    mad_wide_cc_u32 (u0, mu1, te, u0);
    madc_wide_cc_u32(u1, mu3, te, u1);
    madc_wide_cc_u32(u2, mu5, te, u2);
    madc_wide_cc_u32(u3, mu7, te, u3);
    addc_u64        (u4, 0, 0);       

    mad_wide_cc_u32 (q0, mu1, tf, q0);
    madc_wide_cc_u32(q1, mu3, tf, q1);
    madc_wide_cc_u32(q2, mu5, tf, q2);
    madc_wide_cc_u32(q3, mu7, tf, q3);
    addc_u64        (q4, 0, 0);       


    mad_wide_cc_u32 (u0, mu0, tf, u0);
    madc_wide_cc_u32(u1, mu2, tf, u1);
    madc_wide_cc_u32(u2, mu4, tf, u2);
    madc_wide_cc_u32(u3, mu6, tf, u3);
    madc_wide_u32   (u4, mu8, tf, u4);

    //////////////////////////////////
    // q += u >> 32
    //////////////////////////////////

    unpack(l0, h0, u0);                           unpack(l1, h1, q0);  add_cc_u32 (h0, h0, l1); pack(q0, l0, h0);
    unpack(l0, h0, u1);  addc_cc_u32(l0, l0, h1); unpack(l1, h1, q1);  addc_cc_u32(h0, h0, l1); pack(q1, l0, h0);
    unpack(l0, h0, u2);  addc_cc_u32(l0, l0, h1); unpack(l1, h1, q2);  addc_cc_u32(h0, h0, l1); pack(q2, l0, h0);
    unpack(l0, h0, u3);  addc_cc_u32(l0, l0, h1); unpack(l1, h1, q3);  addc_cc_u32(h0, h0, l1); pack(q3, l0, h0);
    unpack(l0, h0, u4);  addc_cc_u32(l0, l0, h1); unpack(l1, h1, q4);  addc_cc_u32(h0, h0, l1); pack(q4, l0, h0);

    //////////////////////////////////
    // r = q * m mod 2^288
    //////////////////////////////////

    unpack(l0, h0, q0);

    mul_wide_u32(r0, m0, h0);
    mul_wide_u32(r1, m2, h0);
    mul_wide_u32(r2, m4, h0);
    mul_wide_u32(r3, m6, h0);

    mul_wide_u32(u0, m1, h0);
    mul_wide_u32(u1, m3, h0);
    mul_wide_u32(u2, m5, h0);
    mul_wide_u32(u3, m7, h0);

    unpack(l0, h0, q1);

    mad_wide_cc_u32 (r1, m1, l0, r1);
    madc_wide_cc_u32(r2, m3, l0, r2);
    madc_wide_cc_u32(r3, m5, l0, r3);
    madc_wide_u32   (r4, m7, l0,  0);

    mad_wide_cc_u32 (u0, m0, l0, u0);
    madc_wide_cc_u32(u1, m2, l0, u1);
    madc_wide_cc_u32(u2, m4, l0, u2);
    madc_wide_u32   (u3, m6, l0, u3);

    mad_wide_cc_u32 (r1, m0, h0, r1);
    madc_wide_cc_u32(r2, m2, h0, r2);
    madc_wide_cc_u32(r3, m4, h0, r3);
    madc_wide_u32   (r4, m6, h0, r4);

    mad_wide_cc_u32 (u1, m1, h0, u1);
    madc_wide_cc_u32(u2, m3, h0, u2);
    madc_wide_u32   (u3, m5, h0, u3);

    unpack(l0, h0, q2);

    mad_wide_cc_u32 (r2, m1, l0, r2);
    madc_wide_cc_u32(r3, m3, l0, r3);
    madc_wide_u32   (r4, m5, l0, r4);

    mad_wide_cc_u32 (u1, m0, l0, u1);
    madc_wide_cc_u32(u2, m2, l0, u2);
    madc_wide_u32   (u3, m4, l0, u3);

    mad_wide_cc_u32 (r2, m0, h0, r2);
    madc_wide_cc_u32(r3, m2, h0, r3);
    madc_wide_u32   (r4, m4, h0, r4);

    mad_wide_cc_u32 (u2, m1, h0, u2);
    madc_wide_u32   (u3, m3, h0, u3);

    unpack(l0, h0, q3);

    mad_wide_cc_u32 (r3, m1, l0, r3);
    madc_wide_u32   (r4, m3, l0, r4);

    mad_wide_cc_u32 (u2, m0, l0, u2);
    madc_wide_u32   (u3, m2, l0, u3);

    mad_wide_cc_u32 (r3, m0, h0, r3);
    madc_wide_u32   (r4, m2, h0, r4);

    mad_wide_u32    (u3, m1, h0, u3);

    unpack(l0, h0, q4);

    mad_wide_cc_u32 (r4, m1, l0, r4);

    mad_wide_cc_u32 (u3, m0, l0, u3);

    mad_wide_cc_u32 (r4, m0, h0, r4);

    //////////////////////////////////
    // r += u << 32
    // r %= 1 << 288
    //////////////////////////////////

    unpack(l0, h0, r0);                           unpack(l1, h1, u0);  add_cc_u32 (h0, h0, l1); pack(r0, l0, h0);
    unpack(l0, h0, r1);  addc_cc_u32(l0, l0, h1); unpack(l1, h1, u1);  addc_cc_u32(h0, h0, l1); pack(r1, l0, h0);
    unpack(l0, h0, r2);  addc_cc_u32(l0, l0, h1); unpack(l1, h1, u2);  addc_cc_u32(h0, h0, l1); pack(r2, l0, h0);
    unpack(l0, h0, r3);  addc_cc_u32(l0, l0, h1); unpack(l1, h1, u3);  addc_cc_u32(h0, h0, l1); pack(r3, l0, h0);
    unpack(l0, h0, r4);  addc_cc_u32(l0, l0, h1); unpack(l1, h1, u4);  addc_cc_u32(h0, h0, l1); pack(r4, l0,  0);

    //////////////////////////////////
    // r = (x % (1 << 288)) - r
    //////////////////////////////////

    unpack(l0, h0, r0); sub_cc_u32 (t0, t0, l0); subc_cc_u32(t1, t1, h0); pack(z0, t0, t1);
    unpack(l0, h0, r1); subc_cc_u32(t2, t2, l0); subc_cc_u32(t3, t3, h0); pack(z1, t2, t3);
    unpack(l0, h0, r2); subc_cc_u32(t4, t4, l0); subc_cc_u32(t5, t5, h0); pack(z2, t4, t5);
    unpack(l0, h0, r3); subc_cc_u32(t6, t6, l0); subc_cc_u32(t7, t7, h0); pack(z3, t6, t7);
    unpack(l0, h0, r4); subc_cc_u32(t8, t8, l0);                          pack(r4, t8,  0);

    assert(r4 == 0);
}

#endif
// vim: ts=4 et sw=4 si
