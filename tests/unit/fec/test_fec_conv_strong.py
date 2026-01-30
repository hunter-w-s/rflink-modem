import random
import pytest
import inspect


# --------------------------------------------------------------------------------------
# Adapter: try to locate your conv encoder/decoder with minimal assumptions.
# Edit ONLY this section if needed.
# --------------------------------------------------------------------------------------

def _load_conv_api():
    """
    Returns (encode_bits, decode_bits, kwargs_template).

    encode_bits(bits: list[int], **kwargs) -> list[int]  # encoded bits
    decode_bits(bits: list[int], **kwargs) -> list[int]  # decoded bits

    kwargs_template is a dict of default kwargs if your API requires them.
    """
    # 1) Try module: rflink_modem.protocol.fec.fec_conv
    try:
        from rflink_modem.protocol.fec import fec_conv as m  # type: ignore
    except Exception:
        try:
            from rflink_modem.protocol.fec.fec_conv import (  # type: ignore
                conv_encode_bits as _enc,
                conv_decode_bits as _dec,
            )
            return _enc, _dec, {}
        except Exception as e:
            raise RuntimeError(
                "Could not import conv FEC module. Adjust adapter in test_fec_conv_strong.py"
            ) from e

    # Common function name candidates
    enc_candidates = [
        "conv_encode_bits",
        "fec_conv_encode_bits",
        "conv_encode",
        "encode_conv",
        "encode_bits",
        "encode",
    ]
    dec_candidates = [
        "viterbi_decode_hard",
        "conv_decode_bits",
        "fec_conv_decode_bits",
        "viterbi_decode_bits",
        "conv_decode",
        "decode_conv",
        "decode_bits",
        "decode",
        "viterbi_decode",
    ]


    enc = None
    for name in enc_candidates:
        if hasattr(m, name):
            enc = getattr(m, name)
            break

    dec = None
    for name in dec_candidates:
        if hasattr(m, name):
            dec = getattr(m, name)
            break

    if enc is None or dec is None:
        raise RuntimeError(
            f"Could not find conv encode/decode functions in {m.__name__}. "
            f"Found: {[a for a in dir(m) if 'enc' in a or 'dec' in a or 'viterbi' in a]}"
        )

    kwargs = dict(
        constraint_len=7,
        polynomials=(0o171, 0o133),
        tail=True,          # or False if you use tail-biting
        traceback=35,       # if your decoder exposes it
    )
    return enc, dec, kwargs


ENCODE_BITS, DECODE_BITS, DEFAULT_KWARGS = _load_conv_api()


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

def _rand_bits(rng: random.Random, n: int) -> list[int]:
    return [rng.getrandbits(1) for _ in range(n)]


def _flip_positions(bits: list[int], positions: list[int]) -> list[int]:
    out = bits[:]
    for p in positions:
        out[p] ^= 1
    return out


def _flip_random(rng: random.Random, bits: list[int], n_flips: int) -> list[int]:
    n = len(bits)
    if n == 0 or n_flips <= 0:
        return bits[:]
    positions = rng.sample(range(n), min(n_flips, n))
    return _flip_positions(bits, positions)


def _flip_burst(bits: list[int], start: int, length: int) -> list[int]:
    out = bits[:]
    n = len(out)
    for i in range(length):
        p = start + i
        if 0 <= p < n:
            out[p] ^= 1
    return out

def _filtered_kwargs(fn, kwargs: dict) -> dict:
    """Return only kwargs that `fn` actually accepts."""
    sig = inspect.signature(fn)
    params = sig.parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs  # fn accepts **kwargs
    return {k: v for k, v in kwargs.items() if k in params}



def _encode(bits: list[int], **kwargs) -> list[int]:
    cw = ENCODE_BITS(bits, **_filtered_kwargs(ENCODE_BITS, kwargs))
    return list(cw)


def _decode(bits: list[int], **kwargs) -> list[int]:
    msg = DECODE_BITS(bits, **_filtered_kwargs(DECODE_BITS, kwargs))
    return list(msg)

def _bit_errors(a: list[int], b: list[int]) -> int:
    return sum((x ^ y) & 1 for x, y in zip(a, b))



# --------------------------------------------------------------------------------------
# Strong tests
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("n_bits", [0, 1, 2, 7, 8, 9, 31, 32, 33, 127, 128, 129, 511])
def test_conv_roundtrip_clean(n_bits):
    rng = random.Random(0xC0DE0000 + n_bits)
    msg = _rand_bits(rng, n_bits)

    cw = _encode(msg, **DEFAULT_KWARGS)
    out = _decode(cw, msg_len=len(msg), **DEFAULT_KWARGS)


    assert out == msg



def test_conv_deterministic_and_pure():
    rng = random.Random(0xBEEF)
    msg = _rand_bits(rng, 200)

    cw1 = _encode(msg, **DEFAULT_KWARGS)
    cw2 = _encode(msg, **DEFAULT_KWARGS)

    assert cw1 == cw2, "encoder must be deterministic"

    cw_copy = cw1[:]
    _ = _decode(cw1, **DEFAULT_KWARGS)
    assert cw1 == cw_copy, "decoder must not mutate input"


@pytest.mark.parametrize("trial_seed", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("n_bits", [32, 128, 512, 1024])
def test_conv_corrects_light_random_noise(trial_seed, n_bits):
    """
    Convolutional code should correct some random bit flips at reasonable rates.
    This does NOT try to define the hard limit; it just guards against regressions
    like wrong polynomials, broken traceback, or wrong tail handling.
    """
    rng = random.Random(0x515151 + trial_seed * 10000 + n_bits)
    msg = _rand_bits(rng, n_bits)
    cw = _encode(msg, **DEFAULT_KWARGS)

    # Flip ~0.5% of encoded bits (light noise)
    n_flips = max(1, len(cw) // 200)
    cw_bad = _flip_random(rng, cw, n_flips)

    out = _decode(cw_bad, **DEFAULT_KWARGS)
    assert out == msg, f"failed with {n_flips} flips over {len(cw)} bits"


def _bit_errors(a: list[int], b: list[int]) -> int:
    return sum(((x ^ y) & 1) for x, y in zip(a, b))


@pytest.mark.parametrize("n_bits", [128, 512, 1024])
def test_conv_burst_noise_degrades_gracefully(n_bits):
    """
    Hard-decision Viterbi is not guaranteed to perfectly correct a contiguous burst
    on the encoded stream. This test is a regression guard: it must not catastrophically
    fail (high BER) under a moderate burst.
    """
    rng = random.Random(0x1234 + n_bits)
    msg = _rand_bits(rng, n_bits)
    cw = _encode(msg, **DEFAULT_KWARGS)

    # Burst length: ~1% of cw (capped)
    burst = min(max(5, len(cw) // 100), 60)
    start = rng.randrange(0, max(1, len(cw) - burst))
    cw_bad = _flip_burst(cw, start, burst)

    out = _decode(cw_bad, **DEFAULT_KWARGS)
    assert len(out) == len(msg)

    ber = _bit_errors(out, msg) / max(1, len(msg))

    # Tune this threshold as you like; start realistic for hard-decision, no interleaver.
    assert ber <= 0.02, f"BER too high {ber:.3%} start={start} len={burst} cw_len={len(cw)}"



def test_conv_many_trials_smoke():
    """
    Quick randomized stress test with modest noise.
    Keep this cheap so the whole suite stays fast.
    """
    rng = random.Random(0xA11CE)
    for k in range(50):
        n_bits = rng.choice([24, 80, 200, 600])
        msg = _rand_bits(rng, n_bits)
        cw = _encode(msg, **DEFAULT_KWARGS)

        # 0 to ~1% flips
        n_flips = rng.randrange(0, max(1, len(cw) // 100))
        cw_bad = _flip_random(rng, cw, n_flips)

        out = _decode(cw_bad, **DEFAULT_KWARGS)
        assert out == msg, f"trial={k} n_bits={n_bits} flips={n_flips} cw_len={len(cw)}"


@pytest.mark.parametrize("n_bits", [64, 256])
def test_conv_fails_gracefully_under_heavy_noise(n_bits):
    """
    Convolutional decode often always returns *something* (no inherent uncorrectable flag).
    This test ensures that under very heavy corruption you don't crash, and output length matches input.
    """
    rng = random.Random(0xDEAD + n_bits)
    msg = _rand_bits(rng, n_bits)
    cw = _encode(msg, **DEFAULT_KWARGS)

    # Flip ~20% of encoded bits (very heavy)
    n_flips = max(1, len(cw) // 5)
    cw_bad = _flip_random(rng, cw, n_flips)

    out = _decode(cw_bad, **DEFAULT_KWARGS)
    assert len(out) == len(msg), "decoded message length must match original length"
    # no assert equality here; it's expected to fail sometimes under heavy noise


def test_conv_non_bits_are_handled_consistently():
    """
    Define the contract: encoder/decoder treat inputs as bits via (x & 1).
    This avoids depending on input validation behavior.
    """
    msg = [0, 1, 0, 2, 1, -3, 7]
    msg_norm = [x & 1 for x in msg]

    cw1 = _encode(msg, **DEFAULT_KWARGS)
    cw2 = _encode(msg_norm, **DEFAULT_KWARGS)
    assert cw1 == cw2

