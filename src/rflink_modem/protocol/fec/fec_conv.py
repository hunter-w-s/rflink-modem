# src/rflink_modem/protocol/fec_conv.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple


def _parity(x: int) -> int:
    # parity of bits in x
    return bin(x).count("1") & 1


@dataclass(frozen=True, slots=True)
class ConvParams:
    K: int = 7
    g1: int = 0o171
    g2: int = 0o133


def conv_encode_bits(bits: Sequence[int], *, params: ConvParams = ConvParams(), tail: bool = True) -> List[int]:
    """
    Convolutional encode (rate 1/2). Returns a flat list of output bits [b0,b1,b2,...].
    Convention:
      - shift register is K bits including current input bit as LSB of the updated register output-bit source.
      - we implement state as K-bit register updated by: state = (state<<1 | input) & ((1<<K)-1)
      - output bits are parity(state & g1), parity(state & g2)
    """
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


def viterbi_decode_hard(rx: Sequence[int], *, params: ConvParams = ConvParams(), tail: bool = True) -> List[int]:
    """
    Hard-decision Viterbi for the same code as conv_encode_bits().

    rx: sequence of 0/1 bits of even length (pairs per input bit).

    Returns decoded input bits. If tail=True, strips the K-1 tail bits.
    """
    if len(rx) % 2 != 0:
        raise ValueError("rx length must be even")
    K = params.K
    n_states = 1 << (K - 1)

    # Precompute trellis: for each state and input bit, the next state and output pair
    trellis_next = [[0, 0] for _ in range(n_states)]
    trellis_out = [[(0, 0), (0, 0)] for _ in range(n_states)]  # [state][input] -> (o1,o2)

    for s in range(n_states):
        for inp in (0, 1):
            full = ((s << 1) | inp) & ((1 << K) - 1)
            o1 = _parity(full & params.g1)
            o2 = _parity(full & params.g2)
            ns = full & ((1 << (K - 1)) - 1)  # drop oldest bit, keep K-1 bits
            trellis_next[s][inp] = ns
            trellis_out[s][inp] = (o1, o2)

    # Viterbi DP
    INF = 10**9
    metric = [INF] * n_states
    metric[0] = 0  # start in zero state
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
                # Hamming distance for hard-decision
                dist = (o1 ^ r1) + (o2 ^ r2)
                m = metric[s] + dist
                if m < new_metric[ns]:
                    new_metric[ns] = m
                    bs[ns] = s
                    bb[ns] = inp

        metric = new_metric
        back_state.append(bs)
        back_bit.append(bb)

    # End state: if tail=True, should end at 0; else choose best
    end_state = 0 if tail else min(range(n_states), key=lambda s: metric[s])

    # Traceback
    decoded: List[int] = []
    s = end_state
    for i in range(n_steps - 1, -1, -1):
        inp = back_bit[i][s]
        ps = back_state[i][s]
        if inp < 0 or ps < 0:
            # unreachable path
            raise ValueError("Viterbi traceback failed")
        decoded.append(inp)
        s = ps

    decoded.reverse()

    if tail:
        # strip K-1 tail bits
        if len(decoded) < (K - 1):
            return []
        decoded = decoded[: -(K - 1)]

    return decoded
