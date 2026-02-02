from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

# ---- RS core copied/adapted from previous version ----
# NOTE: Keep this file self-contained as a module implementation.

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


def _poly_mul(p: List[int], q: List[int]) -> List[int]:
    r = [0] * (len(p) + len(q) - 1)
    for i, a in enumerate(p):
        if a == 0:
            continue
        for j, b in enumerate(q):
            if b == 0:
                continue
            r[i + j] ^= _gf_mul(a, b)
    return r


def _poly_eval(poly: List[int], x: int) -> int:
    y = 0
    for c in poly:
        y = _gf_mul(y, x) ^ c
    return y


def _rs_generator_poly(nsym: int) -> List[int]:
    g = [1]
    for i in range(1, nsym + 1):
        g = _poly_mul(g, [1, _gf_pow(2, i)])
    return g


def rs_encode(msg: bytes, *, nsym: int) -> bytes:
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


# --- Decoder (ascending-polynomial convention) ---

def _poly_add_asc(p: List[int], q: List[int]) -> List[int]:
    r = [0] * max(len(p), len(q))
    for i in range(len(p)):
        r[i] ^= p[i]
    for i in range(len(q)):
        r[i] ^= q[i]
    return r


def _poly_eval_asc(poly: List[int], x: int) -> int:
    y = 0
    xp = 1
    for a in poly:
        if a:
            y ^= _gf_mul(a, xp)
        xp = _gf_mul(xp, x)
    return y


def rs_decode(codeword: bytes, *, nsym: int) -> bytes:
    if nsym <= 0:
        raise ValueError("nsym must be > 0")
    if len(codeword) > 255:
        raise ValueError("codeword length must be <= 255")
    if len(codeword) <= nsym:
        raise ValueError("codeword too short for nsym")

    cw = list(codeword)
    n = len(cw)
    fcr = 1

    synd: List[int] = []
    for i in range(nsym):
        synd.append(_poly_eval(cw, _gf_pow(2, i + fcr)))

    if max(synd) == 0:
        return bytes(cw[:-nsym])

    lam: List[int] = [1]
    B: List[int] = [1]
    L = 0
    m = 1
    b = 1

    for r in range(nsym):
        d = synd[r]
        for i in range(1, L + 1):
            d ^= _gf_mul(lam[i], synd[r - i])

        if d == 0:
            m += 1
            continue

        T = lam[:]
        coef = _gf_div(d, b)

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

    err_pos: List[int] = []
    for p in range(n):
        i = n - 1 - p
        exp = (i + fcr - 1) % 255
        x = _gf_pow(2, (255 - exp) % 255)
        if _poly_eval_asc(lam, x) == 0:
            err_pos.append(p)

    if len(err_pos) != (len(lam) - 1):
        raise ValueError("could not locate all errors")

    t = len(err_pos)
    a_list = [_gf_pow(2, fcr + i) for i in range(t)]
    A: List[List[int]] = [[0] * t for _ in range(t)]
    bvec: List[int] = synd[:t]

    for i, ai in enumerate(a_list):
        for j, p in enumerate(err_pos):
            power = (n - 1 - p) % 255
            A[i][j] = _gf_pow(ai, power)

    for col in range(t):
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

        inv_p = _gf_inverse(A[col][col])
        for c in range(col, t):
            A[col][c] = _gf_mul(A[col][c], inv_p)
        bvec[col] = _gf_mul(bvec[col], inv_p)

        for r in range(col + 1, t):
            factor = A[r][col]
            if factor == 0:
                continue
            for c in range(col, t):
                A[r][c] ^= _gf_mul(factor, A[col][c])
            bvec[r] ^= _gf_mul(factor, bvec[col])

    e = [0] * t
    for r in range(t - 1, -1, -1):
        acc = bvec[r]
        for c in range(r + 1, t):
            acc ^= _gf_mul(A[r][c], e[c])
        e[r] = acc

    for p, mag in zip(err_pos, e):
        cw[p] ^= mag

    synd2: List[int] = []
    for i in range(nsym):
        synd2.append(_poly_eval(cw, _gf_pow(2, i + fcr)))

    if max(synd2) != 0:
        raise ValueError("uncorrectable RS block")

    return bytes(cw[:-nsym])


# ---- Normalized module surface ----

@dataclass(frozen=True)
class Config:
    """
    RS(255, 255-nsym) block code operating on bytes.

    nsym: parity bytes per block (corrects up to nsym//2 byte errors per block).
    k: message bytes per block = 255 - nsym
    pad: padding byte used to fill final partial block to length k.
         rx returns padded message length; trimming belongs to framing/stage later.
    """
    nsym: int = 32
    pad: int = 0x00


def tx(data: bytes, *, cfg: Any) -> bytes:
    """
    Encode arbitrary-length data into concatenated RS codewords.
    Each codeword is 255 bytes: k data + nsym parity.
    """
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError("tx: data must be bytes-like")

    nsym = _get_nsym(cfg)
    pad = _get_pad(cfg)
    k = 255 - nsym

    b = bytes(data)
    if len(b) == 0:
        return b

    out = bytearray()
    for i in range(0, len(b), k):
        block = b[i:i + k]
        if len(block) < k:
            block = block + bytes([pad]) * (k - len(block))
        out += rs_encode(block, nsym=nsym)
    return bytes(out)


def rx(data: bytes, *, cfg: Any) -> bytes:
    """
    Decode concatenated 255-byte RS codewords back into message bytes.
    Returns padded length (multiple of k).
    Raises ValueError if any block is uncorrectable.
    """
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError("rx: data must be bytes-like")

    nsym = _get_nsym(cfg)
    k = 255 - nsym

    b = bytes(data)
    if len(b) == 0:
        return b
    if len(b) % 255 != 0:
        raise ValueError(f"rx: length {len(b)} not divisible by 255 (RS codeword size)")

    out = bytearray()
    for i in range(0, len(b), 255):
        cw = b[i:i + 255]
        out += rs_decode(cw, nsym=nsym)
    # out length is multiple of k (padded). Trimming later.
    if len(out) % k != 0:
        raise AssertionError("internal: decoded length not multiple of k")
    return bytes(out)


def _get_nsym(cfg: Any) -> int:
    nsym = getattr(cfg, "nsym", None)
    if nsym is None:
        raise AttributeError("cfg missing required attribute: nsym")
    if not isinstance(nsym, int):
        raise TypeError("cfg.nsym must be int")
    if not (1 <= nsym <= 254):
        raise ValueError("cfg.nsym must be in [1,254]")
    return nsym


def _get_pad(cfg: Any) -> int:
    pad = getattr(cfg, "pad", None)
    if pad is None:
        raise AttributeError("cfg missing required attribute: pad")
    if not isinstance(pad, int):
        raise TypeError("cfg.pad must be int")
    if not (0 <= pad <= 255):
        raise ValueError("cfg.pad must be in [0,255]")
    return pad
