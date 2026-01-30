# tests/unit/test_fec_rs_strong.py

import random
import pytest

from rflink_modem.protocol.fec.fec_rs import rs_encode, rs_decode


def _corrupt_bytes(rng: random.Random, cw: bytes, positions, *, allow_zero_delta=False) -> bytes:
    """Return a new codeword with byte errors applied at given positions."""
    out = bytearray(cw)
    for p in positions:
        # xor with non-zero delta by default
        delta = rng.randrange(256) if allow_zero_delta else rng.randrange(1, 256)
        out[p] ^= delta
    return bytes(out)


@pytest.mark.parametrize("nsym", [6, 10, 16, 32])
@pytest.mark.parametrize("msg_len", [1, 2, 5, 17, 50, 120])
def test_rs_roundtrip_many_sizes_and_nsym(nsym, msg_len):
    rng = random.Random(0xC0FFEE + nsym * 1000 + msg_len)
    msg = bytes(rng.randrange(256) for _ in range(msg_len))
    cw = rs_encode(msg, nsym=nsym)
    out = rs_decode(cw, nsym=nsym)
    assert out == msg


@pytest.mark.parametrize("nsym", [6, 10, 16, 32])
@pytest.mark.parametrize("msg_len", [1, 5, 50, 120])
def test_rs_corrects_up_to_t_errors_random_trials(nsym, msg_len):
    # Keep runtime sane but still meaningful
    trials = 50 if nsym <= 16 else 30
    t = nsym // 2

    base_seed = 0xBADC0DE + nsym * 100 + msg_len
    for k in range(trials):
        rng = random.Random(base_seed + k)

        msg = bytes(rng.randrange(256) for _ in range(msg_len))
        cw = rs_encode(msg, nsym=nsym)

        positions = rng.sample(range(len(cw)), t)
        cw_bad = _corrupt_bytes(rng, cw, positions)

        out = rs_decode(cw_bad, nsym=nsym)
        assert out == msg, f"failed nsym={nsym} msg_len={msg_len} positions={sorted(positions)}"


@pytest.mark.parametrize("nsym", [6, 10, 16, 32])
@pytest.mark.parametrize("msg_len", [5, 50, 120])
def test_rs_fails_for_more_than_t_errors(nsym, msg_len):
    # Correctness contract: > t byte errors should raise
    rng = random.Random(0xDEADBEEF + nsym * 1000 + msg_len)
    t = nsym // 2

    msg = bytes(rng.randrange(256) for _ in range(msg_len))
    cw = rs_encode(msg, nsym=nsym)

    positions = rng.sample(range(len(cw)), t + 1)
    cw_bad = _corrupt_bytes(rng, cw, positions)

    with pytest.raises(ValueError):
        rs_decode(cw_bad, nsym=nsym)


@pytest.mark.parametrize("nsym", [10, 16])
@pytest.mark.parametrize("msg_len", [50, 120])
def test_rs_edge_positions_and_parity_region(nsym, msg_len):
    # Deterministic "nasty" positions: start, end, and parity-only
    rng = random.Random(0x123456 + nsym * 1000 + msg_len)
    t = nsym // 2

    msg = bytes(rng.randrange(256) for _ in range(msg_len))
    cw = rs_encode(msg, nsym=nsym)
    n = len(cw)

    # Case A: include first and last byte
    positions_a = {0, n - 1}
    # fill remaining errors deterministically
    remaining = [i for i in range(n) if i not in positions_a]
    positions_a |= set(remaining[: max(0, t - len(positions_a))])
    positions_a = sorted(list(positions_a))[:t]

    out_a = rs_decode(_corrupt_bytes(rng, cw, positions_a), nsym=nsym)
    assert out_a == msg, f"edge-pos correction failed positions={positions_a}"

    # Case B: parity-only errors (last nsym bytes)
    positions_b = list(range(n - nsym, n))[:t]
    out_b = rs_decode(_corrupt_bytes(rng, cw, positions_b), nsym=nsym)
    assert out_b == msg, f"parity-only correction failed positions={positions_b}"


@pytest.mark.parametrize("nsym", [10, 16])
def test_rs_duplicate_corruption_same_index_equivalent(nsym):
    # If the same byte is "corrupted twice" (XOR twice), it should behave correctly.
    rng = random.Random(0xFACE + nsym)
    msg = bytes(rng.randrange(256) for _ in range(50))
    cw = rs_encode(msg, nsym=nsym)

    t = nsym // 2
    positions = rng.sample(range(len(cw)), t)

    # Apply corruption once
    rng1 = random.Random(0x1111)
    cw_bad_once = _corrupt_bytes(rng1, cw, positions)

    # Apply two corruptions at same positions with same deltas => cancels (net no error)
    rng2 = random.Random(0x2222)
    deltas = [rng2.randrange(1, 256) for _ in positions]
    out = bytearray(cw)
    for p, d in zip(positions, deltas):
        out[p] ^= d
    for p, d in zip(positions, deltas):
        out[p] ^= d

    cw_canceled = bytes(out)

    # once should decode to msg, canceled should decode to msg with no correction needed
    assert rs_decode(cw_bad_once, nsym=nsym) == msg
    assert rs_decode(cw_canceled, nsym=nsym) == msg


def test_rs_rejects_invalid_lengths():
    rng = random.Random(0xABC)
    msg = bytes(rng.randrange(256) for _ in range(10))
    cw = rs_encode(msg, nsym=10)

    with pytest.raises(ValueError):
        rs_decode(cw[:10], nsym=10)  # too short: len <= nsym
