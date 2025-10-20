#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bit-exact IEEE-754 binary16 (fp16) multiply — integer & rounding correct.

Fix vs previous: subnormal rounding now uses the RAW exponent sum (eua+eub),
not the post-normalization exponent. This removes the ±1 ulp / ×2 errors you saw.
"""

import os, struct, random

# ---- fp16 constants ----
EXP_BITS = 5
FRAC_BITS = 10
EXP_BIAS = 15

SIGN_MASK = 0x8000
EXP_MASK  = 0x7C00
FRAC_MASK = 0x03FF

CANON_QNAN = 0x7E00  # canonical quiet NaN (payload ignored for equality)

def bits_to_fields(h):
    s = (h >> 15) & 1
    e = (h >> 10) & 0x1F
    f = h & FRAC_MASK
    return s, e, f

def is_nan(h):  s,e,f = bits_to_fields(h); return e == 0x1F and f != 0
def is_inf(h):  s,e,f = bits_to_fields(h); return e == 0x1F and f == 0
def is_zero(h): s,e,f = bits_to_fields(h); return e == 0 and f == 0
def sign_bit(h): return (h >> 15) & 1

# ---- fp16 <-> float (for reference / testing only) ----
def fp16_to_float(h):
    return struct.unpack('>e', h.to_bytes(2,'big'))[0]

def float_to_fp16_bits_safe(x):
    try:
        return int.from_bytes(struct.pack('>e', x), 'big')
    except OverflowError:
        s = 1 if x < 0 else 0
        return (s << 15) | 0x7C00

# ---- exact integer multiply, IEEE-754 compliant ----
def mul_fp16_bits_exact(a_bits: int, b_bits: int) -> int:
    # Special cases (IEEE-754)
    if is_nan(a_bits) or is_nan(b_bits):
        return CANON_QNAN

    a_inf, b_inf = is_inf(a_bits), is_inf(b_bits)
    a_zero, b_zero = is_zero(a_bits), is_zero(b_bits)

    s = sign_bit(a_bits) ^ sign_bit(b_bits)

    # 0 * Inf = NaN
    if (a_inf and b_zero) or (b_inf and a_zero):
        return CANON_QNAN

    # Inf * finite = Inf
    if a_inf or b_inf:
        return (s << 15) | 0x7C00

    # 0 * finite = signed zero
    if a_zero or b_zero:
        return (s << 15)

    # Decode fields
    _, ea, fa = bits_to_fields(a_bits)
    _, eb, fb = bits_to_fields(b_bits)

    # Normalize significands and get unbiased exponents
    # Represent value as: value = sig * 2^(e_un - 10), where sig is 11-bit with top bit at position 10
    def norm_sig_exp(e, f):
        if e == 0:
            # subnormal: value = f * 2^(-24)
            # normalize f to 11-bit range [2^10 .. 2^11-1] and adjust exponent
            h = f.bit_length() - 1  # highest bit (0..9) since f in [1..1023]
            shift_left = 10 - h     # bring msb to position 10
            sig = f << shift_left
            e_un = (1 - EXP_BIAS) - shift_left  # = -14 - (10 - h) = h - 24
            return sig, e_un
        else:
            sig = (1 << 10) | f
            e_un = e - EXP_BIAS
            return sig, e_un

    sig_a, eua = norm_sig_exp(ea, fa)
    sig_b, eub = norm_sig_exp(eb, fb)

    # 11-bit × 11-bit product
    P = sig_a * sig_b                 # < 2^22
    e_sum_raw = eua + eub             # RAW unbiased exponent sum (keep for subnormal path)

    # Normalize product to 11-bit mantissa
    # P is in [2^20 .. <2^22). If bit21 is set -> shift 11, else shift 10
    if (P & (1 << 21)) != 0:
        shift = 11
    else:
        shift = 10

    mant = P >> shift                 # 11-bit pre-rounded mantissa
    rem  = P & ((1 << shift) - 1)     # remainder bits below mant
    E_un_norm = e_sum_raw + (shift - 10)

    # Rounding-to-nearest-even helper for (mant, rem) right shift by 'shift'
    def round_nearest_even(m, r, shift_amt):
        if shift_amt == 0:
            return m, 0
        guard  = (r >> (shift_amt - 1)) & 1
        sticky = 1 if (r & ((1 << (shift_amt - 1)) - 1)) != 0 else 0
        inc = 1 if (guard and (sticky or (m & 1))) else 0
        m2 = m + inc
        carry = 1 if m2 == (1 << 11) else 0
        if carry:
            m2 >>= 1
        return m2, carry

    # Check for overflow to infinity (after rounding of a normal)
    exp_biased = E_un_norm + EXP_BIAS

    # If the normal exponent is well above range, it's Inf
    # (We still do the rounding step first; carry may bump exponent.)
    # First, perform rounding in the normal path only if we stay normal.
    if exp_biased >= 1:
        # Perform normal rounding (uses the remainder from normalization)
        m2, carry = round_nearest_even(mant, rem, shift)
        E_un2 = E_un_norm + carry
        exp_biased2 = E_un2 + EXP_BIAS
        if exp_biased2 >= 0x1F:
            return (s << 15) | 0x7C00  # ±Inf
        # If still normal
        if exp_biased2 >= 1:
            frac = m2 & ((1 << FRAC_BITS) - 1)
            return (s << 15) | ((exp_biased2 & 0x1F) << 10) | frac
        # If rounding pushed us down to subnormal (rare), fall through to subnormal path below with E_un2
        E_un_norm = E_un2
        exp_biased = exp_biased2

    # Subnormal/underflow region: exp_biased <= 0
    # Use RAW exponent sum formula: frac = round_even( P * 2^(e_sum_raw + 4) )
    # Because exact value = P * 2^(e_sum_raw - 20); subnormal value = frac * 2^(-24).
    # Equate: frac ≈ P * 2^(e_sum_raw + 4).
    if exp_biased <= 0:
        shift_amt = -(e_sum_raw + 4)  # must be >= 1 for subnormal
        if shift_amt > 40:
            frac_sub = 0  # too small → zero
        else:
            base = P >> shift_amt
            guard = (P >> (shift_amt - 1)) & 1
            sticky = 1 if (P & ((1 << (shift_amt - 1)) - 1)) != 0 else 0
            inc = 1 if (guard and (sticky or (base & 1))) else 0
            frac_sub = base + inc

        if frac_sub >= (1 << FRAC_BITS):
            # Rounded up into the smallest normal (exp=1, frac=0)
            return (s << 15) | (1 << 10)
        else:
            # True subnormal (exp field = 0)
            return (s << 15) | frac_sub

    # If we got here, exp_biased==0 (boundary). The subnormal branch above handles it.
    # But for completeness (should not hit), return signed zero.
    return (s << 15)

# ---- reference path (struct) for testing ----
def ref_mul_bits(a_bits: int, b_bits: int) -> int:
    a = fp16_to_float(a_bits)
    b = fp16_to_float(b_bits)
    return float_to_fp16_bits_safe(a * b)

def fp16_bits_equal(a_bits: int, b_bits: int) -> bool:
    # Treat all NaNs equal for test purposes
    if is_nan(a_bits) and is_nan(b_bits):
        return True
    return a_bits == b_bits

# ---- test harness ----
def quick_self_test(samples: int = 300_000, seed: int = 7) -> None:
    rnd = random.Random(seed)
    corners = [
        0x0000, 0x8000,           # +0, -0
        0x7C00, 0xFC00,           # +Inf, -Inf
        0x7E00, 0x7D00,           # qNaN (canon), sNaN-ish payload
        0x3C00, 0xBC00,           # +1, -1
        0x3555,                   # ~0.333
        0x7BFF,                   # max finite
        0x0400, 0x0001, 0x03FF,   # min normal, min subnormal, max subnormal
    ]
    pairs = [(rnd.getrandbits(16), rnd.getrandbits(16)) for _ in range(samples)]
    for a in corners:
        for b in corners:
            pairs.append((a, b))

    mism = 0
    for i, (a, b) in enumerate(pairs, 1):
        ref = ref_mul_bits(a, b)
        got = mul_fp16_bits_exact(a, b)
        if not fp16_bits_equal(ref, got):
            mism += 1
            if mism <= 30:
                print(f"Mismatch {mism}: a=0x{a:04X}, b=0x{b:04X}, ref=0x{ref:04X}, got=0x{got:04X}")
    total = len(pairs)
    if mism == 0:
        print(f"PASS: {total} cases matched (NaNs treated equal).")
    else:
        print(f"FAIL: {mism}/{total} mismatches.")

if __name__ == "__main__":
    n = int(os.environ.get("MIA_FP16_SAMPLES", "300000"))
    quick_self_test(samples=n)

