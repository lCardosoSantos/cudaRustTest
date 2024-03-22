// Copyright 2024 Dag Arne Osvik
// Copyright 2024 Luan Cardoso dos Santos

#ifndef FR_REDC_CUH
#define FR_REDC_CUH

#include <cassert>
#include <cstdint>
// #include <cstdio>

#include "fr.cuh"
#include "ptx.cuh"

__forceinline__ __device__ void fr_redc(
    uint64_t &z0,
    uint64_t &z1,
    uint64_t &z2,
    uint64_t &z3,
    uint64_t  z4,
    uint64_t  z5,
    uint64_t  z6,
    uint64_t  z7)
{
    const uint32_t
        m0  = 0xF0000001,
        m1  = 0x43E1F593,
        m2  = 0x79B97091,
        m3  = 0x2833E848,
        m4  = 0x8181585D,
        m5  = 0xB85045B6,
        m6  = 0xE131A029,
        m7  = 0x30644E72,
        inv = 0xEFFFFFFF;

    uint64_t
        t;

    uint32_t
        mul,    // multiplier to use for each reduction step
        tl, th,
        t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, ta, tb, tc, td, te, tf,
        c8, c9, ca, cb, cc, cd, ce, cf;

        c8 = c9 = ca = cb = cc = cd = ce = cf = 0;

#ifndef NDEBUG
    printf("<%s>\n", __func__);

    printf("0: #x%016lx%016lx%016lx%016lx%016lx%016lx%016lx%016lx\n", z7, z6, z5, z4, z3, z2, z1, z0);
#endif

    unpack(t0, t1, z0);
    unpack(t2, t3, z1);
    unpack(t4, t5, z2);
    unpack(t6, t7, z3);
    unpack(t8, t9, z4);
    unpack(ta, tb, z5);
    unpack(tc, td, z6);
    unpack(te, tf, z7);

    assert(tf <= m7);    // redc only works for z <= 2^256 * m

    mul_lo_u32(mul, inv, t0);

    mul_wide_u32(t, mul, m0); unpack(tl, th, t); add_cc_u32 (t0, t0, tl); addc_cc_u32(t1, t1, th);
    mul_wide_u32(t, mul, m2); unpack(tl, th, t); addc_cc_u32(t2, t2, tl); addc_cc_u32(t3, t3, th);
    mul_wide_u32(t, mul, m4); unpack(tl, th, t); addc_cc_u32(t4, t4, tl); addc_cc_u32(t5, t5, th);
    mul_wide_u32(t, mul, m6); unpack(tl, th, t); addc_cc_u32(t6, t6, tl); addc_cc_u32(t7, t7, th); addc_u32(c8, 0, 0);

    mul_wide_u32(t, mul, m1); unpack(tl, th, t); add_cc_u32 (t1, t1, tl); addc_cc_u32(t2, t2, th);
    mul_wide_u32(t, mul, m3); unpack(tl, th, t); addc_cc_u32(t3, t3, tl); addc_cc_u32(t4, t4, th);
    mul_wide_u32(t, mul, m5); unpack(tl, th, t); addc_cc_u32(t5, t5, tl); addc_cc_u32(t6, t6, th);
    mul_wide_u32(t, mul, m7); unpack(tl, th, t); addc_cc_u32(t7, t7, tl); addc_cc_u32(t8, t8, th); addc_u32(c9, 0, 0);

    // printf("0: #x%08x%08x%08x%08x%08x%08x%08x%08x%08x%08x%08x%08x%08x%08x%08x%08x\n", tf, te, td, tc, tb, ta, t9+c9, t8+c8, t7, t6, t5, t4, t3, t2, t1, t0);

    mul_lo_u32(mul, inv, t1);

    mul_wide_u32(t, mul, m0); unpack(tl, th, t); add_cc_u32 (t1, t1, tl); addc_cc_u32(t2, t2, th);
    mul_wide_u32(t, mul, m2); unpack(tl, th, t); addc_cc_u32(t3, t3, tl); addc_cc_u32(t4, t4, th);
    mul_wide_u32(t, mul, m4); unpack(tl, th, t); addc_cc_u32(t5, t5, tl); addc_cc_u32(t6, t6, th);
    mul_wide_u32(t, mul, m6); unpack(tl, th, t); addc_cc_u32(t7, t7, tl); addc_cc_u32(t8, t8, th); addc_u32(c9, 0, c9);

    mul_wide_u32(t, mul, m1); unpack(tl, th, t); add_cc_u32 (t2, t2, tl); addc_cc_u32(t3, t3, th);
    mul_wide_u32(t, mul, m3); unpack(tl, th, t); addc_cc_u32(t4, t4, tl); addc_cc_u32(t5, t5, th);
    mul_wide_u32(t, mul, m5); unpack(tl, th, t); addc_cc_u32(t6, t6, tl); addc_cc_u32(t7, t7, th);
    mul_wide_u32(t, mul, m7); unpack(tl, th, t); addc_cc_u32(t8, t8, tl); addc_cc_u32(t9, t9, th); addc_u32(ca, 0, 0);

    // printf("1: #x%08x%08x%08x%08x%08x%08x%08x%08x%08x%08x%08x%08x%08x%08x%08x%08x\n", tf, te, td, tc, tb, ta+ca, t9+c9, t8+c8, t7, t6, t5, t4, t3, t2, t1, t0);

    mul_lo_u32(mul, inv, t2);

    mul_wide_u32(t, mul, m0); unpack(tl, th, t); add_cc_u32 (t2, t2, tl); addc_cc_u32(t3, t3, th);
    mul_wide_u32(t, mul, m2); unpack(tl, th, t); addc_cc_u32(t4, t4, tl); addc_cc_u32(t5, t5, th);
    mul_wide_u32(t, mul, m4); unpack(tl, th, t); addc_cc_u32(t6, t6, tl); addc_cc_u32(t7, t7, th);
    mul_wide_u32(t, mul, m6); unpack(tl, th, t); addc_cc_u32(t8, t8, tl); addc_cc_u32(t9, t9, th); addc_u32(ca, 0, ca);

    mul_wide_u32(t, mul, m1); unpack(tl, th, t); add_cc_u32 (t3, t3, tl); addc_cc_u32(t4, t4, th);
    mul_wide_u32(t, mul, m3); unpack(tl, th, t); addc_cc_u32(t5, t5, tl); addc_cc_u32(t6, t6, th);
    mul_wide_u32(t, mul, m5); unpack(tl, th, t); addc_cc_u32(t7, t7, tl); addc_cc_u32(t8, t8, th);
    mul_wide_u32(t, mul, m7); unpack(tl, th, t); addc_cc_u32(t9, t9, tl); addc_cc_u32(ta, ta, th); addc_u32(cb, 0, 0);

    // printf("2: #x%08x%08x%08x%08x%08x%08x%08x%08x%08x%08x%08x%08x%08x%08x%08x%08x\n", tf, te, td, tc, tb+cb, ta+ca, t9+c9, t8+c8, t7, t6, t5, t4, t3, t2, t1, t0);

    mul_lo_u32(mul, inv, t3);

    mul_wide_u32(t, mul, m0); unpack(tl, th, t); add_cc_u32 (t3, t3, tl); addc_cc_u32(t4, t4, th);
    mul_wide_u32(t, mul, m2); unpack(tl, th, t); addc_cc_u32(t5, t5, tl); addc_cc_u32(t6, t6, th);
    mul_wide_u32(t, mul, m4); unpack(tl, th, t); addc_cc_u32(t7, t7, tl); addc_cc_u32(t8, t8, th);
    mul_wide_u32(t, mul, m6); unpack(tl, th, t); addc_cc_u32(t9, t9, tl); addc_cc_u32(ta, ta, th); addc_u32(cb, 0, cb);

    mul_wide_u32(t, mul, m1); unpack(tl, th, t); add_cc_u32 (t4, t4, tl); addc_cc_u32(t5, t5, th);
    mul_wide_u32(t, mul, m3); unpack(tl, th, t); addc_cc_u32(t6, t6, tl); addc_cc_u32(t7, t7, th);
    mul_wide_u32(t, mul, m5); unpack(tl, th, t); addc_cc_u32(t8, t8, tl); addc_cc_u32(t9, t9, th);
    mul_wide_u32(t, mul, m7); unpack(tl, th, t); addc_cc_u32(ta, ta, tl); addc_cc_u32(tb, tb, th); addc_u32(cc, 0, 0);

    // printf("3: #x%08x%08x%08x%08x%08x%08x%08x%08x%08x%08x%08x%08x%08x%08x%08x%08x\n", tf+cf, te+ce, td+cd, tc+cc, tb+cb, ta+ca, t9+c9, t8+c8, t7, t6, t5, t4, t3, t2, t1, t0);

    mul_lo_u32(mul, inv, t4);

    mul_wide_u32(t, mul, m0); unpack(tl, th, t); add_cc_u32 (t4, t4, tl); addc_cc_u32(t5, t5, th);
    mul_wide_u32(t, mul, m2); unpack(tl, th, t); addc_cc_u32(t6, t6, tl); addc_cc_u32(t7, t7, th);
    mul_wide_u32(t, mul, m4); unpack(tl, th, t); addc_cc_u32(t8, t8, tl); addc_cc_u32(t9, t9, th);
    mul_wide_u32(t, mul, m6); unpack(tl, th, t); addc_cc_u32(ta, ta, tl); addc_cc_u32(tb, tb, th); addc_u32(cc, 0, cc);

    mul_wide_u32(t, mul, m1); unpack(tl, th, t); add_cc_u32 (t5, t5, tl); addc_cc_u32(t6, t6, th);
    mul_wide_u32(t, mul, m3); unpack(tl, th, t); addc_cc_u32(t7, t7, tl); addc_cc_u32(t8, t8, th);
    mul_wide_u32(t, mul, m5); unpack(tl, th, t); addc_cc_u32(t9, t9, tl); addc_cc_u32(ta, ta, th);
    mul_wide_u32(t, mul, m7); unpack(tl, th, t); addc_cc_u32(tb, tb, tl); addc_cc_u32(tc, tc, th); addc_u32(cd, 0, 0);

    // printf("4: #x%08x%08x%08x%08x%08x%08x%08x%08x%08x%08x%08x%08x%08x%08x%08x%08x\n", tf+cf, te+ce, td+cd, tc+cc, tb+cb, ta+ca, t9+c9, t8+c8, t7, t6, t5, t4, t3, t2, t1, t0);

    mul_lo_u32(mul, inv, t5);

    mul_wide_u32(t, mul, m0); unpack(tl, th, t); add_cc_u32 (t5, t5, tl); addc_cc_u32(t6, t6, th);
    mul_wide_u32(t, mul, m2); unpack(tl, th, t); addc_cc_u32(t7, t7, tl); addc_cc_u32(t8, t8, th);
    mul_wide_u32(t, mul, m4); unpack(tl, th, t); addc_cc_u32(t9, t9, tl); addc_cc_u32(ta, ta, th);
    mul_wide_u32(t, mul, m6); unpack(tl, th, t); addc_cc_u32(tb, tb, tl); addc_cc_u32(tc, tc, th); addc_u32(cd, 0, cd);

    mul_wide_u32(t, mul, m1); unpack(tl, th, t); add_cc_u32 (t6, t6, tl); addc_cc_u32(t7, t7, th);
    mul_wide_u32(t, mul, m3); unpack(tl, th, t); addc_cc_u32(t8, t8, tl); addc_cc_u32(t9, t9, th);
    mul_wide_u32(t, mul, m5); unpack(tl, th, t); addc_cc_u32(ta, ta, tl); addc_cc_u32(tb, tb, th);
    mul_wide_u32(t, mul, m7); unpack(tl, th, t); addc_cc_u32(tc, tc, tl); addc_cc_u32(td, td, th); addc_u32(ce, 0, 0);

    // printf("5: #x%08x%08x%08x%08x%08x%08x%08x%08x%08x%08x%08x%08x%08x%08x%08x%08x\n", tf+cf, te+ce, td+cd, tc+cc, tb+cb, ta+ca, t9+c9, t8+c8, t7, t6, t5, t4, t3, t2, t1, t0);

    mul_lo_u32(mul, inv, t6);

    mul_wide_u32(t, mul, m0); unpack(tl, th, t); add_cc_u32 (t6, t6, tl); addc_cc_u32(t7, t7, th);
    mul_wide_u32(t, mul, m2); unpack(tl, th, t); addc_cc_u32(t8, t8, tl); addc_cc_u32(t9, t9, th);
    mul_wide_u32(t, mul, m4); unpack(tl, th, t); addc_cc_u32(ta, ta, tl); addc_cc_u32(tb, tb, th);
    mul_wide_u32(t, mul, m6); unpack(tl, th, t); addc_cc_u32(tc, tc, tl); addc_cc_u32(td, td, th); addc_u32(ce, 0, ce);

    mul_wide_u32(t, mul, m1); unpack(tl, th, t); add_cc_u32 (t7, t7, tl); addc_cc_u32(t8, t8, th);
    mul_wide_u32(t, mul, m3); unpack(tl, th, t); addc_cc_u32(t9, t9, tl); addc_cc_u32(ta, ta, th);
    mul_wide_u32(t, mul, m5); unpack(tl, th, t); addc_cc_u32(tb, tb, tl); addc_cc_u32(tc, tc, th);
    mul_wide_u32(t, mul, m7); unpack(tl, th, t); addc_cc_u32(td, td, tl); addc_cc_u32(te, te, th); addc_u32(cf, 0, 0);

    // printf("6: #x%08x%08x%08x%08x%08x%08x%08x%08x%08x%08x%08x%08x%08x%08x%08x%08x\n", tf+cf, te+ce, td+cd, tc+cc, tb+cb, ta+ca, t9+c9, t8+c8, t7, t6, t5, t4, t3, t2, t1, t0);

    mul_lo_u32(mul, inv, t7);

    mul_wide_u32(t, mul, m0); unpack(tl, th, t); add_cc_u32 (t7, t7, tl); addc_cc_u32(t8, t8, th);
    mul_wide_u32(t, mul, m2); unpack(tl, th, t); addc_cc_u32(t9, t9, tl); addc_cc_u32(ta, ta, th);
    mul_wide_u32(t, mul, m4); unpack(tl, th, t); addc_cc_u32(tb, tb, tl); addc_cc_u32(tc, tc, th);
    mul_wide_u32(t, mul, m6); unpack(tl, th, t); addc_cc_u32(td, td, tl); addc_cc_u32(te, te, th); addc_u32(cf, 0, cf);

    mul_wide_u32(t, mul, m1); unpack(tl, th, t); add_cc_u32 (t8, t8, tl); addc_cc_u32(t9, t9, th);
    mul_wide_u32(t, mul, m3); unpack(tl, th, t); addc_cc_u32(ta, ta, tl); addc_cc_u32(tb, tb, th);
    mul_wide_u32(t, mul, m5); unpack(tl, th, t); addc_cc_u32(tc, tc, tl); addc_cc_u32(td, td, th);
    mul_wide_u32(t, mul, m7); unpack(tl, th, t); addc_cc_u32(te, te, tl); addc_cc_u32(tf, tf, th);

    // printf("7: #x%08x%08x%08x%08x%08x%08x%08x%08x%08x%08x%08x%08x%08x%08x%08x%08x\n", tf+cf, te+ce, td+cd, tc+cc, tb+cb, ta+ca, t9+c9, t8+c8, t7, t6, t5, t4, t3, t2, t1, t0);

    add_cc_u32 (t8, t8, c8);
    addc_cc_u32(t9, t9, c9);
    addc_cc_u32(ta, ta, ca);
    addc_cc_u32(tb, tb, cb);
    addc_cc_u32(tc, tc, cc);
    addc_cc_u32(td, td, cd);
    addc_cc_u32(te, te, ce);
    addc_u32   (tf, tf, cf);

    // printf("8: #x%08x%08x%08x%08x%08x%08x%08x%08x%08x%08x%08x%08x%08x%08x%08x%08x\n", tf, te, td, tc, tb, ta, t9, t8, t7, t6, t5, t4, t3, t2, t1, t0);

    pack(z0, t8, t9);
    pack(z1, ta, tb);
    pack(z2, tc, td);
    pack(z3, te, tf);

    // printf("9: #x%016lx%016lx%016lx%016lx\n", z3, z2, z1, z0);
}

#endif

// vim: ts=4 et sw=4 si
