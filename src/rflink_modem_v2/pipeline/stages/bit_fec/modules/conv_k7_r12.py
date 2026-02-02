from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Sequence

# Reference core (ported from previous version) :contentReference[oaicite:1]{index=1}


def _parity(x: int) -> int:
    return bin(x).count("1") & 1


@dataclass(frozen=True, slots=True)
class _ConvParams:
    K: int = 7
    g1: int = 0o171
    g2: int = 0o133


def _conv_encode_bits(bits: Sequence[int], *, params: _ConvParams, tail: bool) -> List[int]:
    K = params.K
    state = 0
    out: List[int] = []

    seq = list(bits)
    if tail:
        seq.extend([0] * (K - 1))

    for b in seq:
        state = ((state << 1) | (b & 1)) & ((1 << K) - 1)
        out.append(_parity(state & params.g1))
        out.append(_parity(state & params.g2))

    return out


def _viterbi_decode_hard(rx: Sequence[int], *, params: _ConvParams, tail: bool) -> List[int]:
    if len(rx) % 2 != 0:
        raise ValueError("rx bit length must be even (rate 1/2 pairs)")

    K = params.K
    n_states = 1 << (K - 1)

    trellis_next = [[0, 0] for _ in range(n_states)]
    trellis_out = [[(0, 0), (0, 0)] for _ in range(n_states)]

    for s in range(n_states):
        for inp in (0, 1):
            full = ((s << 1) | inp) & ((1 << K) - 1)
            o1 = _parity(full & params.g1)
            o2 = _parity(full & params.g2)
            ns = full & ((1 << (K - 1)) - 1)
            trellis_next[s][inp] = ns
            trellis_out[s][inp] = (o1, o2)

    INF = 10**9
    metric = [INF] * n_states
    metric[0] = 0
    back_state: List[List[int]] = []
    back_bit: List[List[int]] = []

    n_steps = len(rx) // 2
    for i in range(n_steps):
        r1 = rx[2 * i] & 1
        r2 = rx[2 * i + 1] & 1

        new_metric = [INF] * n_states
        bs = [-1] * n_states
        bb = [-1] * n_states

        for s in range(n_states):
            if metric[s] >= INF:
                continue
            for inp in (0, 1):
                ns = trellis_next[s][inp]
                o1, o2 = trellis_out[s][inp]
                dist = (o1 ^ r1) + (o2 ^ r2)
                m = metric[s] + dist
                if m < new_metric[ns]:
                    new_metric[ns] = m
                    bs[ns] = s
                    bb[ns] = inp

        metric = new_metric
        back_state.append(bs)
        back_bit.append(bb)

    end_state = 0 if tail else min(range(n_states), key=lambda s: metric[s])

    decoded: List[int] = []
    s = end_state
    for i in range(n_steps - 1, -1, -1):
        inp = back_bit[i][s]
        ps = back_state[i][s]
        if inp < 0 or ps < 0:
            raise ValueError("Viterbi traceback failed")
        decoded.append(inp)
        s = ps

    decoded.reverse()

    if tail:
        if len(decoded) < (K - 1):
            return []
        decoded = decoded[: -(K - 1)]

    return decoded


# ----------------------------
# Normalized module surface
# ----------------------------

@dataclass(frozen=True)
class Config:
    """
    Bit-level convolutional FEC (rate 1/2) with hard-decision Viterbi.

    Input/Output of tx/rx are BYTES, interpreted as a packed bitstream.

    Bit order:
      - bytes are expanded MSB-first: bit7, bit6, ..., bit0
      - encoded bits are packed MSB-first into output bytes

    tail:
      - if True, encoder appends K-1 zeros; decoder assumes end state 0 and strips K-1 bits
      - if False, decoder selects best end state (no tail stripping)

    K/g1/g2 default to the common K=7, (171,133) octal code.
    """
    K: int = 7
    g1: int = 0o171
    g2: int = 0o133
    tail: bool = True


def tx(data: bytes, *, cfg: Any) -> bytes:
    """
    Encode packed bits (from data bytes) into packed bits (bytes).
    """
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError("tx: data must be bytes-like")

    params = _get_params(cfg)
    tail = _get_tail(cfg)

    bits = _bytes_to_bits_msb(bytes(data))
    enc_bits = _conv_encode_bits(bits, params=params, tail=tail)
    return _bits_to_bytes_msb(enc_bits)


def rx(data: bytes, *, cfg: Any) -> bytes:
    """
    Decode packed encoded bits (bytes) back into packed original bits (bytes).
    """
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError("rx: data must be bytes-like")

    params = _get_params(cfg)
    tail = _get_tail(cfg)

    rx_bits = _bytes_to_bits_msb(bytes(data))

    # CRITICAL: remove the bit-padding that tx() adds to byte-align the encoded stream.
    pad_bits = _enc_pad_bits(K=params.K, tail=tail)
    if pad_bits:
        if len(rx_bits) < pad_bits:
            raise ValueError("rx: encoded stream too short")
        rx_bits = rx_bits[:-pad_bits]

    # Now rx_bits length should be even for rate-1/2 pairs.
    if len(rx_bits) % 2 != 0:
        raise ValueError("rx: encoded bit length not even after trimming padding")

    dec_bits = _viterbi_decode_hard(rx_bits, params=params, tail=tail)

    # For byte-input, decoded bit count should be a multiple of 8.
    if len(dec_bits) % 8 != 0:
        raise ValueError("rx: decoded bit length not byte-aligned")

    return _bits_to_bytes_msb(dec_bits)


# ----------------------------
# Packing helpers
# ----------------------------

def _bytes_to_bits_msb(b: bytes) -> List[int]:
    out: List[int] = []
    for x in b:
        for i in range(7, -1, -1):
            out.append((x >> i) & 1)
    return out


def _bits_to_bytes_msb(bits: Sequence[int]) -> bytes:
    n = len(bits)
    if n == 0:
        return b""
    pad = (-n) % 8
    if pad:
        bits = list(bits) + [0] * pad

    out = bytearray(len(bits) // 8)
    k = 0
    for bi in range(0, len(bits), 8):
        v = 0
        for j in range(8):
            v = (v << 1) | (bits[bi + j] & 1)
        out[k] = v
        k += 1
    return bytes(out)

def _enc_pad_bits(*, K: int, tail: bool) -> int:
    """
    Number of bit-padding zeros tx() adds at the *end* of the encoded stream
    to make it byte-aligned.

    For byte-input (8*L bits):
      - tail=False: encoded bits = 16*L => already byte-aligned => 0 pad bits
      - tail=True:  encoded bits = 16*L + 2*(K-1)
                   pad bits = (-2*(K-1)) mod 8  (independent of L)
    """
    if not tail:
        return 0
    return (-2 * (K - 1)) % 8



def _get_params(cfg: Any) -> _ConvParams:
    K = getattr(cfg, "K", None)
    g1 = getattr(cfg, "g1", None)
    g2 = getattr(cfg, "g2", None)

    for name, v in (("K", K), ("g1", g1), ("g2", g2)):
        if v is None:
            raise AttributeError(f"cfg missing required attribute: {name}")
        if not isinstance(v, int):
            raise TypeError(f"cfg.{name} must be int")

    if K < 2 or K > 15:
        raise ValueError("cfg.K must be in [2,15] for practicality")
    if g1 <= 0 or g2 <= 0:
        raise ValueError("cfg.g1 and cfg.g2 must be > 0")

    return _ConvParams(K=K, g1=g1, g2=g2)


def _get_tail(cfg: Any) -> bool:
    tail = getattr(cfg, "tail", None)
    if tail is None:
        raise AttributeError("cfg missing required attribute: tail")
    if not isinstance(tail, bool):
        raise TypeError("cfg.tail must be bool")
    return tail
