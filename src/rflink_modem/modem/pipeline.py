# src/rflink_modem/modem/pipeline.py
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json
import numpy as np

from rflink_modem.containers.wav import write_wav_mono
from rflink_modem.protocol.fec.fec_rs import rs_encode, rs_decode
from rflink_modem.protocol.fec.interleave import interleave_bytes, deinterleave_bytes
from rflink_modem.protocol.framing import pack_frame, unpack_frame, HEADER_LEN
from rflink_modem.protocol.fec.fec_conv import conv_encode_bits, viterbi_decode_hard
from rflink_modem.modem.audio.afsk_tx import bits_to_pcm, AFSKTxConfig
from rflink_modem.modem.audio.afsk_rx import demod_pcm_to_bits


# ----------------------------
# Bit/byte helpers
# ----------------------------

def _bytes_to_bits(data: bytes) -> list[int]:
    out: list[int] = []
    for b in data:
        for i in range(8):
            out.append((b >> (7 - i)) & 1)
    return out


def _bits_to_bytes(bits: list[int]) -> bytes:
    if len(bits) % 8 != 0:
        raise ValueError(f"bit length must be multiple of 8 (got {len(bits)})")
    out = bytearray(len(bits) // 8)
    k = 0
    for i in range(len(out)):
        v = 0
        for _ in range(8):
            v = (v << 1) | (bits[k] & 1)
            k += 1
        out[i] = v
    return bytes(out)


def _pack_bits(bits: list[int]) -> bytes:
    """Pack bits (MSB-first) into bytes for debug dumps."""
    b = bytearray((len(bits) + 7) // 8)
    for i, bit in enumerate(bits):
        if bit & 1:
            b[i // 8] |= 1 << (7 - (i % 8))
    return bytes(b)


# ----------------------------
# Debug dump helpers
# ----------------------------

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _dump_bytes(debug_dir: Path | None, name: str, data: bytes) -> None:
    if debug_dir is None:
        return
    _ensure_dir(debug_dir)
    (debug_dir / name).write_bytes(data)


def _dump_text(debug_dir: Path | None, name: str, text: str) -> None:
    if debug_dir is None:
        return
    _ensure_dir(debug_dir)
    (debug_dir / name).write_text(text, encoding="utf-8")


def _dump_bits(debug_dir: Path | None, name: str, bits: list[int]) -> None:
    if debug_dir is None:
        return
    _dump_bytes(debug_dir, name, _pack_bits(bits))


def _dump_npy(debug_dir: Path | None, name: str, arr: np.ndarray) -> None:
    if debug_dir is None:
        return
    _ensure_dir(debug_dir)
    np.save(str(debug_dir / name), arr)


# ----------------------------
# Pipeline
# ----------------------------

def tx_bytes_to_pcm(
    payload: bytes,
    *,
    afsk_cfg: AFSKTxConfig,
    nsym: int = 10,
    interleave_depth: int = 8,
    debug_dir: str | Path | None = None,
    verbose: bool = False,
) -> np.ndarray:
    """
    Full TX pipeline: bytes -> PCM samples

    If debug_dir is provided, writes stage artifacts to that folder.
    """
    dbg = Path(debug_dir) if debug_dir is not None else None

    if verbose and dbg is not None:
        _dump_text(
            dbg,
            "manifest_tx.json",
            json.dumps(
                {
                    "direction": "tx",
                    "nsym": nsym,
                    "interleave_depth": interleave_depth,
                    "afsk_cfg": asdict(afsk_cfg),
                    "payload_len": len(payload),
                },
                indent=2,
                sort_keys=True,
            ),
        )

    _dump_bytes(dbg, "00_payload.bin", payload)

    # Frame (includes optional whitening inside framing)
    framed = pack_frame(payload, whiten_payload=True, whiten_seed=0xACE1)
    _dump_bytes(dbg, "01_frame.bin", framed)

    # RS on bytes (frame bytes)
    rs_cw = rs_encode(framed, nsym=nsym)
    _dump_bytes(dbg, "02_rs_codeword.bin", rs_cw)

    # Interleave bytes
    il = interleave_bytes(rs_cw, depth=interleave_depth)
    _dump_bytes(dbg, "03_interleaved.bin", il)

    # Bytes -> bits
    bits = _bytes_to_bits(il)
    _dump_bits(dbg, "04_bits_in.packed", bits)

    # Convolutional encode
    coded_bits = list(conv_encode_bits(bits))
    _dump_bits(dbg, "05_conv_bits.packed", coded_bits)

    # AFSK modulate to PCM
    pcm = bits_to_pcm(coded_bits, afsk_cfg)
    _dump_npy(dbg, "06_pcm.npy", pcm.astype(np.float32, copy=False))
    if dbg is not None:
        write_wav_mono(str(dbg / "06_pcm.wav"), pcm, afsk_cfg.sample_rate)

    return pcm


def rx_pcm_to_bytes(
    pcm: np.ndarray,
    *,
    afsk_cfg: AFSKTxConfig,
    nsym: int = 10,
    interleave_depth: int = 8,
    debug_dir: str | Path | None = None,
    verbose: bool = False,
) -> bytes:
    """
    Full RX pipeline: PCM samples -> bytes

    If debug_dir is provided, writes stage artifacts to that folder.
    """
    dbg = Path(debug_dir) if debug_dir is not None else None

    if verbose and dbg is not None:
        _dump_text(
            dbg,
            "manifest_rx.json",
            json.dumps(
                {
                    "direction": "rx",
                    "nsym": nsym,
                    "interleave_depth": interleave_depth,
                    "afsk_cfg": asdict(afsk_cfg),
                    "pcm_len": int(pcm.shape[0]),
                },
                indent=2,
                sort_keys=True,
            ),
        )

    # Save the incoming PCM too (useful if the test corrupts it)
    _dump_npy(dbg, "00_pcm_in.npy", pcm.astype(np.float32, copy=False))
    if dbg is not None:
        write_wav_mono(str(dbg / "00_pcm_in.wav"), pcm, afsk_cfg.sample_rate)

    # AFSK demod: PCM -> hard bits
    bits = list(demod_pcm_to_bits(pcm, afsk_cfg))
    _dump_bits(dbg, "01_demod_bits.packed", bits)

    # Viterbi hard decode
    decoded_bits = list(viterbi_decode_hard(bits))
    _dump_bits(dbg, "02_viterbi_bits.packed", decoded_bits)

    # Bits -> bytes (interleaved bytes)
    il_bytes = _bits_to_bytes(decoded_bits)
    _dump_bytes(dbg, "03_interleaved_bytes.bin", il_bytes)

    # Deinterleave
    # NOTE: interleave_bytes pads to a rectangle; the *true* RS codeword length is:
    # len(frame) + nsym. We can recover len(frame) by deinterleaving just the header
    # and parsing payload_len. Easiest robust approach:
    #
    # 1) Deinterleave using the padded length first (no trim)
    # 2) RS-decode will validate and framing CRC will validate
    #
    # However, deinterleave_bytes currently requires the original_len (pre-pad length).
    # We can compute that by first deinterleaving enough bytes to read HEADER_LEN.
    #
    # Strategy: deinterleave_bytes with original_len=len(il_bytes) (no trim) to get a
    # header candidate, parse payload_len from it, then compute the true RS length.
    #
    # Because your current HEADER is at the beginning of the frame, this works.

    deint_full = deinterleave_bytes(il_bytes, depth=interleave_depth, original_len=len(il_bytes))
    _dump_bytes(dbg, "04_deinterleaved_full.bin", deint_full)

    # Parse header to compute the true frame length and thus RS length.
    # unpack_frame expects a full frame; we don’t have that yet.
    # But HEADER_LEN gives us fixed header bytes, and payload_len is stored in the header.
    if len(deint_full) < HEADER_LEN:
        raise ValueError("deinterleaved data too short to contain header")

    # Extract payload_len from the header bytes:
    # framing.py stores payload_len as uint16 little-endian at bytes 4..6 (per your implementation).
    # To avoid baking offsets here, we instead RS decode first using the *maximum* plausible length.
    #
    # Robust approach without needing header offsets:
    # Try RS decode on the deinterleaved_full as-is; if padding exists, RS decode will fail.
    #
    # Therefore, we instead compute the *true* RS codeword length using the frame CRC scheme:
    # We can attempt to RS-decode progressively shorter suffix trims until success, but keep it cheap.
    #
    # For now, we do a bounded search trimming up to depth*cols padding (<= interleave_depth).
    #
    rs_bytes = None
    trim_max = interleave_depth  # padding is at most depth-1 columns worth; conservative cap
    for trim in range(0, trim_max + 1):
        candidate = deint_full[:-trim] if trim > 0 else deint_full
        try:
            rs_bytes = candidate
            frame_bytes = rs_decode(rs_bytes, nsym=nsym)
            _hdr, payload = unpack_frame(frame_bytes)
            # success
            _dump_bytes(dbg, "05_rs_input.bin", rs_bytes)
            _dump_bytes(dbg, "06_frame_out.bin", frame_bytes)
            _dump_bytes(dbg, "07_payload_out.bin", payload)
            return payload
        except Exception:
            continue

    raise ValueError("RX pipeline failed: could not RS-decode/de-frame after deinterleave (trim search exhausted)")
