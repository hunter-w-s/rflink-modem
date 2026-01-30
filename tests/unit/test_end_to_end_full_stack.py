import random
import math
import inspect
import pytest
from rflink_modem.modem.audio.afsk_tx import AFSKTxConfig

# -------------------------
# Small utilities
# -------------------------

def _filtered_kwargs(fn, kwargs: dict) -> dict:
    sig = inspect.signature(fn)
    params = sig.parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in params}

def _find_attr(mod, names):
    for n in names:
        if hasattr(mod, n):
            return getattr(mod, n)
    return None

def _flip_bits_in_place(bits, positions):
    for p in positions:
        if 0 <= p < len(bits):
            bits[p] ^= 1

def _rand_bytes(rng: random.Random, n: int) -> bytes:
    return bytes(rng.randrange(256) for _ in range(n))

def _bytes_to_bits(data: bytes) -> list[int]:
    out = []
    for b in data:
        for i in range(8):
            out.append((b >> (7 - i)) & 1)
    return out

def _bits_to_bytes(bits: list[int]) -> bytes:
    if len(bits) % 8 != 0:
        raise ValueError("bit length must be multiple of 8")
    out = bytearray(len(bits) // 8)
    k = 0
    for i in range(len(out)):
        v = 0
        for _ in range(8):
            v = (v << 1) | (bits[k] & 1)
            k += 1
        out[i] = v
    return bytes(out)

# -------------------------
# Adapters to your codebase
# -------------------------

def _load_full_stack():
    """
    Returns a dict of callables:
      framing: frame_encode(bytes)->bytes, frame_decode(bytes)->bytes
      whitening: whiten(bytes)->bytes, dewhiten(bytes)->bytes
      rs: rs_encode(bytes, nsym)->bytes, rs_decode(bytes, nsym)->bytes
      interleave: interleave_bytes(bytes, depth)->bytes, deinterleave_bytes(bytes, depth, original_len)->bytes
      conv: conv_encode_bits(list[int])->list[int], viterbi_decode_hard(list[int])->list[int]
      afsk: afsk_modulate(bits)->samples, afsk_demodulate(samples)->bits

    If something can’t be found, we skip.
    """
    stack = {}

    # --- Framing ---
    try:
        from rflink_modem.protocol import framing as framing_mod  # type: ignore
    except Exception:
        framing_mod = None

    if framing_mod:
        frame_enc = _find_attr(framing_mod, ["frame_encode", "encode_frame", "build_frame", "make_frame"])
        frame_dec = _find_attr(framing_mod, ["frame_decode", "decode_frame", "parse_frame", "read_frame"])
        if frame_enc and frame_dec:
            stack["frame_enc"] = frame_enc
            stack["frame_dec"] = frame_dec

    # --- Whitening ---
    try:
        from rflink_modem.protocol import whitening as wh_mod  # type: ignore
    except Exception:
        wh_mod = None

    if wh_mod:
        whiten = _find_attr(wh_mod, ["whiten", "whiten_bytes", "apply_whitening"])
        dewhiten = _find_attr(wh_mod, ["dewhiten", "dewhiten_bytes", "remove_whitening"])
        if whiten and dewhiten:
            stack["whiten"] = whiten
            stack["dewhiten"] = dewhiten

    # --- RS ---
    from rflink_modem.protocol.fec.fec_rs import rs_encode, rs_decode  # type: ignore
    stack["rs_encode"] = rs_encode
    stack["rs_decode"] = rs_decode

    # --- Interleave ---
    try:
        from rflink_modem.protocol.fec.interleave import interleave_bytes, deinterleave_bytes  # type: ignore
        stack["interleave_bytes"] = interleave_bytes
        stack["deinterleave_bytes"] = deinterleave_bytes
    except Exception:
        pass

    # --- Convolutional ---
    from rflink_modem.protocol.fec import fec_conv as conv_mod  # type: ignore
    enc_bits = _find_attr(conv_mod, ["conv_encode_bits"])
    dec_bits = _find_attr(conv_mod, ["viterbi_decode_hard"])
    if enc_bits and dec_bits:
        stack["conv_encode_bits"] = enc_bits
        stack["viterbi_decode_hard"] = dec_bits

    # --- AFSK ---
    try:
        from rflink_modem.modem.audio import afsk_tx as atx
        from rflink_modem.modem.audio import afsk_rx as arx
    except Exception:
        atx = arx = None

    # --- AFSK (PCM-domain) ---
    if atx and arx:
        if hasattr(atx, "bits_to_pcm") and hasattr(arx, "demod_pcm_to_bits"):
            stack["afsk_mod"] = atx.bits_to_pcm
            stack["afsk_demod"] = arx.demod_pcm_to_bits


    return stack

STACK = _load_full_stack()

# -------------------------
# The actual end-to-end test
# -------------------------

@pytest.mark.parametrize("payload_len", [1, 16, 64, 128])
def test_full_stack_end_to_end(payload_len):
    """
    Full-stack trial:
      payload -> framing -> whitening -> RS -> interleave -> conv -> AFSK -> channel -> AFSK -> conv -> deinterleave -> RS -> dewhiten -> deframe
    """
    required = ["rs_encode", "rs_decode", "conv_encode_bits", "viterbi_decode_hard", "afsk_mod", "afsk_demod"]
    missing = [k for k in required if k not in STACK]
    if missing:
        pytest.skip(f"Full-stack test skipped (missing APIs): {missing}")

    rng = random.Random(0x5150 + payload_len)

    afsk_cfg = AFSKTxConfig(
    sample_rate=48000,
    symbol_rate=1200,
    mark_hz=2200.0,
    space_hz=1200.0,
    amplitude=0.8,
    lead_silence_s=0.25,
    trail_silence_s=0.25,
    )


    # Tunables (match your real deployment defaults as you lock them in)
    nsym = 10
    depth = 8
    conv_kwargs = dict(
        # include common keys; we’ll filter per-function signature
        K=7,
        polys=(0o171, 0o133),
        generators=(0o171, 0o133),
        traceback=35,
        tail=True,
    )

    # 1) payload
    payload = _rand_bytes(rng, payload_len)

    # 2) framing (optional if your stack exposes it)
    framed = payload
    if "frame_enc" in STACK and "frame_dec" in STACK:
        framed = STACK["frame_enc"](payload)

    # 3) whitening (optional)
    whitened = framed
    if "whiten" in STACK and "dewhiten" in STACK:
        whitened = STACK["whiten"](framed)

    # 4) RS
    rs_cw = STACK["rs_encode"](whitened, nsym=nsym)

    # 5) interleave bytes (optional but strongly recommended)
    interleaved = rs_cw
    if "interleave_bytes" in STACK:
        interleaved = STACK["interleave_bytes"](rs_cw, depth=depth)

    # 6) bytes -> bits
    bits_in = _bytes_to_bits(interleaved)

    # 7) conv encode
    conv_enc = STACK["conv_encode_bits"]
    coded_bits = list(conv_enc(bits_in, **_filtered_kwargs(conv_enc, conv_kwargs)))

    # 8) AFSK modulate
    samples = STACK["afsk_mod"](coded_bits, afsk_cfg)
    demod_bits = list(STACK["afsk_demod"](samples, afsk_cfg))


    # Random flips ~0.3%
    n_flips = max(1, len(demod_bits) // 300)
    positions = rng.sample(range(len(demod_bits)), n_flips)
    _flip_bits_in_place(demod_bits, positions)

    # A burst of 20 bits somewhere
    burst_len = 20
    start = rng.randrange(0, max(1, len(demod_bits) - burst_len))
    _flip_bits_in_place(demod_bits, list(range(start, start + burst_len)))

    # 10) conv decode
    vdec = STACK["viterbi_decode_hard"]
    decoded_bits = list(vdec(demod_bits, **_filtered_kwargs(vdec, conv_kwargs)))

    # 11) bits -> bytes
    decoded_bytes = _bits_to_bytes(decoded_bits)

    # 12) deinterleave
    deinterleaved = decoded_bytes
    if "deinterleave_bytes" in STACK:
        # interleaver pads to rectangular length; we know what it should be from RS cw
        deinterleaved = STACK["deinterleave_bytes"](decoded_bytes, depth=depth, original_len=len(rs_cw))

    # 13) RS decode
    out_whitened = STACK["rs_decode"](deinterleaved, nsym=nsym)

    # 14) dewhiten
    out_framed = out_whitened
    if "dewhiten" in STACK:
        out_framed = STACK["dewhiten"](out_whitened)

    # 15) deframe
    out_payload = out_framed
    if "frame_dec" in STACK:
        out_payload = STACK["frame_dec"](out_framed)

    assert out_payload == payload
