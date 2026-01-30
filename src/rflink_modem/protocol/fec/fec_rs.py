# src/rflink_modem/protocol/fec_rs.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple


# GF(256) with primitive polynomial 0x11D (x^8 + x^4 + x^3 + x^2 + 1)
_PRIM = 0x11D

_GF_EXP = [0] * 512
_GF_LOG = [0] * 256

_x = 1
for i in range(255):
    _GF_EXP[i] = _x
    _GF_LOG[_x] = i
    _x <<= 1
    if _x & 0x100:
        _x ^= _PRIM
for i in range(255, 512):
    _GF_EXP[i] = _GF_EXP[i - 255]


def _gf_mul(a: int, b: int) -> int:
    if a == 0 or b == 0:
        return 0
    return _GF_EXP[_GF_LOG[a] + _GF_LOG[b]]


def _gf_div(a: int, b: int) -> int:
    if b == 0:
        raise ZeroDivisionError("GF division by zero")
    if a == 0:
        return 0
    return _GF_EXP[(_GF_LOG[a] - _GF_LOG[b]) % 255]


def _gf_pow(a: int, p: int) -> int:
    if p == 0:
        return 1
    if a == 0:
        return 0
    return _GF_EXP[(_GF_LOG[a] * p) % 255]


def _gf_inverse(a: int) -> int:
    if a == 0:
        raise ZeroDivisionError("GF inverse of zero")
    return _GF_EXP[255 - _GF_LOG[a]]


def _poly_add(p: Sequence[int], q: Sequence[int]) -> List[int]:
    r = [0] * max(len(p), len(q))
    # align right
    for i in range(len(p)):
        r[i + len(r) - len(p)] ^= p[i]
    for i in range(len(q)):
        r[i + len(r) - len(q)] ^= q[i]
    return r


def _poly_mul(p: Sequence[int], q: Sequence[int]) -> List[int]:
    r = [0] * (len(p) + len(q) - 1)
    for i, a in enumerate(p):
        if a == 0:
            continue
        for j, b in enumerate(q):
            if b == 0:
                continue
            r[i + j] ^= _gf_mul(a, b)
    return r


def _poly_eval(poly: Sequence[int], x: int) -> int:
    y = 0
    for c in poly:
        y = _gf_mul(y, x) ^ c
    return y


def _rs_generator_poly(nsym: int) -> List[int]:
    g = [1]
    # Standard choice: roots at alpha^1 .. alpha^nsym
    for i in range(1, nsym + 1):
        g = _poly_mul(g, [1, _gf_pow(2, i)])
    return g



def rs_encode(msg: bytes, *, nsym: int) -> bytes:
    """
    Systematic RS encode: codeword = msg || parity (nsym bytes).
    Typical constraints: len(msg) + nsym <= 255.
    """
    if nsym <= 0:
        raise ValueError("nsym must be > 0")
    if len(msg) + nsym > 255:
        raise ValueError("len(msg) + nsym must be <= 255")

    gen = _rs_generator_poly(nsym)
    res = list(msg) + [0] * nsym

    for i in range(len(msg)):
        coef = res[i]
        if coef != 0:
            for j in range(len(gen)):
                res[i + j] ^= _gf_mul(gen[j], coef)

    parity = bytes(res[-nsym:])
    return msg + parity


# --------------------------------------------------------------------------------------
# Decoder (self-consistent ascending-polynomial implementation)
#
# Conventions:
#   - Generator roots: alpha^1 .. alpha^nsym (FCR=1)
#   - Syndromes: S_i = r(alpha^(i+1)) for i=0..nsym-1
#   - Decoder polynomials are ASCENDING: [a0, a1, ..., aN] = a0 + a1*x + ...
# --------------------------------------------------------------------------------------


def _poly_add_asc(p: Sequence[int], q: Sequence[int]) -> List[int]:
    # ASCENDING polynomial add (coeff-wise XOR).
    # IMPORTANT: do NOT trim trailing zeros here.
    # Berlekamp–Massey tracks the current locator degree (L) separately; trimming
    # can shorten `lam` below L+1 and cause index errors / incorrect updates.
    r = [0] * max(len(p), len(q))
    for i in range(len(p)):
        r[i] ^= p[i]
    for i in range(len(q)):
        r[i] ^= q[i]
    return r


def _poly_mul_asc(p: Sequence[int], q: Sequence[int]) -> List[int]:
    r = [0] * (len(p) + len(q) - 1)
    for i, a in enumerate(p):
        if a == 0:
            continue
        for j, b in enumerate(q):
            if b == 0:
                continue
            r[i + j] ^= _gf_mul(a, b)
    while len(r) > 1 and r[-1] == 0:
        r.pop()
    return r


def _poly_eval_asc(poly: Sequence[int], x: int) -> int:
    # poly = [a0, a1, ..., aN] meaning a0 + a1*x + ...
    y = 0
    xp = 1
    for a in poly:
        if a:
            y ^= _gf_mul(a, xp)
        xp = _gf_mul(xp, x)
    return y


def rs_decode(codeword: bytes, *, nsym: int) -> bytes:
    """Decode a codeword (msg||parity). Returns corrected message bytes.

    This decoder is intentionally self-consistent (ascending-polynomial convention)
    to avoid locator/evaluator/Forney mismatch issues.

    Raises ValueError if the RS block is uncorrectable.
    """
    if nsym <= 0:
        raise ValueError("nsym must be > 0")
    if len(codeword) > 255:
        raise ValueError("codeword length must be <= 255")
    if len(codeword) <= nsym:
        raise ValueError("codeword too short for nsym")

    cw = list(codeword)
    n = len(cw)
    fcr = 1  # first consecutive root: alpha^1 .. alpha^nsym

    # Syndromes: S_i = r(alpha^(i+fcr)), i=0..nsym-1
    synd: List[int] = []
    for i in range(nsym):
        synd.append(_poly_eval(cw, _gf_pow(2, i + fcr)))

    if max(synd) == 0:
        return bytes(cw[:-nsym])

    # Berlekamp–Massey in ASCENDING form: Lambda(x) = 1 + l1*x + ...
    lam: List[int] = [1]
    B: List[int] = [1]
    L = 0
    m = 1
    b = 1

    for r in range(nsym):
        # discrepancy d = S_r + sum_{i=1..L} lam[i]*S_{r-i}
        d = synd[r]
        for i in range(1, L + 1):
            d ^= _gf_mul(lam[i], synd[r - i])

        if d == 0:
            m += 1
            continue

        T = lam[:]
        coef = _gf_div(d, b)

        # lam = lam + coef * x^m * B
        xmb = ([0] * m) + B
        lam = _poly_add_asc(lam, [_gf_mul(coef, c) for c in xmb])

        if 2 * L <= r:
            L = r + 1 - L
            B = T
            b = d
            m = 1
        else:
            m += 1

    if len(lam) == 1:
        raise ValueError("RS decode failed to derive locator")

    # Chien search: find roots of Lambda at x = alpha^(-(i+fcr)),
    # where i is position-from-end (0 at last symbol).
    err_pos: List[int] = []
    for p in range(n):
        i = n - 1 - p
        exp = (i + fcr - 1) % 255
        x = _gf_pow(2, (255 - exp) % 255)  # alpha^(-exp)
        if _poly_eval_asc(lam, x) == 0:
            err_pos.append(p)

    


    if len(err_pos) != (len(lam) - 1):
        raise ValueError("could not locate all errors")

    # At this point we have the error LOCATIONS (err_pos). Rather than rely on a
    # Forney/evaluator convention (easy to get subtly wrong with indexing/FCR),
    # solve the error MAGNITUDES directly from the syndrome equations.
    #
    # Codeword is interpreted (by _poly_eval) as a descending polynomial:
    #   c(x) = c[0] x^(n-1) + c[1] x^(n-2) + ... + c[n-1]
    #
    # An error of magnitude e at position p contributes:
    #   e * a^(n-1-p)
    # to the evaluation at a.
    #
    # Our syndromes are:
    #   S_i = r(a_i),  where a_i = alpha^(fcr+i), i=0..nsym-1
    #
    # Therefore, with known error positions p_j:
    #   S_i = Σ_j e_j * a_i^(n-1-p_j)
    #
    # Solve this t×t linear system over GF(256), where t=len(err_pos).

    t = len(err_pos)
    a_list = [_gf_pow(2, fcr + i) for i in range(t)]  # a_i for i=0..t-1
    # Build matrix A (t×t) and vector b (t)
    A: List[List[int]] = [[0] * t for _ in range(t)]
    bvec: List[int] = synd[:t]

    for i, ai in enumerate(a_list):
        for j, p in enumerate(err_pos):
            power = (n - 1 - p) % 255
            A[i][j] = _gf_pow(ai, power)

    # Gaussian elimination in GF(256)
    # Augment A with bvec
    for col in range(t):
        # find pivot
        pivot = None
        for r in range(col, t):
            if A[r][col] != 0:
                pivot = r
                break
        if pivot is None:
            raise ValueError("could not solve RS magnitudes (singular system)")
        if pivot != col:
            A[col], A[pivot] = A[pivot], A[col]
            bvec[col], bvec[pivot] = bvec[pivot], bvec[col]

        # normalize pivot row
        inv_p = _gf_inverse(A[col][col])
        for c in range(col, t):
            A[col][c] = _gf_mul(A[col][c], inv_p)
        bvec[col] = _gf_mul(bvec[col], inv_p)

        # eliminate below
        for r in range(col + 1, t):
            factor = A[r][col]
            if factor == 0:
                continue
            for c in range(col, t):
                A[r][c] ^= _gf_mul(factor, A[col][c])
            bvec[r] ^= _gf_mul(factor, bvec[col])

    # back substitution
    e = [0] * t
    for r in range(t - 1, -1, -1):
        acc = bvec[r]
        for c in range(r + 1, t):
            acc ^= _gf_mul(A[r][c], e[c])
        e[r] = acc  # A[r][r] is 1

    # apply corrections
    for p, mag in zip(err_pos, e):
        cw[p] ^= mag




    # Verify
    synd2: List[int] = []
    for i in range(nsym):
        synd2.append(_poly_eval(cw, _gf_pow(2, i + fcr)))

    

    if max(synd2) != 0:
        raise ValueError("uncorrectable RS block")

    return bytes(cw[:-nsym])
