from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json
import struct
import numpy as np

from rflink_modem.modem.audio.afsk_tx import AFSKTxConfig
from rflink_modem.modem.pipeline import tx_bytes_to_pcm, rx_pcm_to_bytes
from rflink_modem.protocol.framing import HEADER_LEN


def max_payload_per_frame(*, nsym: int) -> int:
    """
    RS(255) constraint: len(codeword)=len(frame_bytes)+nsym must be <=255.
    frame_bytes = HEADER_LEN + payload_len + 4 (payload_crc32)
    """
    return 255 - nsym - HEADER_LEN - 4


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def tx_bytes_to_pcm_blocks(
    data: bytes,
    *,
    afsk_cfg: AFSKTxConfig,
    nsym: int = 10,
    interleave_depth: int = 8,
    debug_dir: str | Path | None = None,
    verbose: bool = False,
    dump_blocks: bool = False,   # NEW: controls per-block stage dumps
) -> list[np.ndarray]:
    """
    Block mode TX: bytes -> list of PCM blocks.

    - If debug_dir is provided, we write top-level manifests there.
    - If dump_blocks=True, we also create per-block folders and pass debug_dir down
      into the single-block pipeline so every stage can be dumped.
    """
    max_pl = max_payload_per_frame(nsym=nsym)
    if max_pl <= 4:
        raise ValueError("nsym too large for framing overhead")

    total_len = len(data)
    first_data = max_pl - 4

    chunks: list[bytes] = []
    chunks.append(struct.pack(">I", total_len) + data[:first_data])

    off = first_data
    while off < total_len:
        chunks.append(data[off : off + max_pl])
        off += max_pl

    base = Path(debug_dir) if debug_dir is not None else None
    if verbose and base is not None:
        _ensure_dir(base)
        (base / "manifest_blocks_tx.json").write_text(
            json.dumps(
                {
                    "direction": "tx_blocks",
                    "nsym": nsym,
                    "interleave_depth": interleave_depth,
                    "afsk_cfg": asdict(afsk_cfg),
                    "total_len": total_len,
                    "num_blocks": len(chunks),
                    "max_payload_per_frame": max_pl,
                    "dump_blocks": dump_blocks,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    pcms: list[np.ndarray] = []
    for i, chunk in enumerate(chunks):
        per_block_tx_dir = None
        if dump_blocks and base is not None:
            block_dir = base / f"block_{i:04d}" / "tx"
            _ensure_dir(block_dir)
            per_block_tx_dir = block_dir

        pcm = tx_bytes_to_pcm(
            chunk,
            afsk_cfg=afsk_cfg,
            nsym=nsym,
            interleave_depth=interleave_depth,
            debug_dir=per_block_tx_dir,
            verbose=verbose if dump_blocks else False,
        )
        pcms.append(pcm)

    return pcms


def rx_pcm_blocks_to_bytes(
    pcms: list[np.ndarray],
    *,
    afsk_cfg: AFSKTxConfig,
    nsym: int = 10,
    interleave_depth: int = 8,
    debug_dir: str | Path | None = None,
    verbose: bool = False,
    dump_blocks: bool = False,   # NEW: controls per-block stage dumps
) -> bytes:
    """
    Block mode RX: list of PCM blocks -> original bytes.

    - If debug_dir is provided, we write top-level manifests there.
    - If dump_blocks=True, we also create per-block folders and pass debug_dir down
      into the single-block pipeline so every stage can be dumped.
    """
    base = Path(debug_dir) if debug_dir is not None else None
    if verbose and base is not None:
        _ensure_dir(base)
        (base / "manifest_blocks_rx.json").write_text(
            json.dumps(
                {
                    "direction": "rx_blocks",
                    "nsym": nsym,
                    "interleave_depth": interleave_depth,
                    "afsk_cfg": asdict(afsk_cfg),
                    "num_blocks": len(pcms),
                    "dump_blocks": dump_blocks,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    parts: list[bytes] = []
    expected_total: int | None = None

    for i, pcm in enumerate(pcms):
        per_block_rx_dir = None
        if dump_blocks and base is not None:
            block_dir = base / f"block_{i:04d}" / "rx"
            _ensure_dir(block_dir)
            per_block_rx_dir = block_dir

        payload = rx_pcm_to_bytes(
            pcm,
            afsk_cfg=afsk_cfg,
            nsym=nsym,
            interleave_depth=interleave_depth,
            debug_dir=per_block_rx_dir,
            verbose=verbose if dump_blocks else False,
        )

        if i == 0:
            if len(payload) < 4:
                raise ValueError("block 0 payload too short")
            expected_total = struct.unpack(">I", payload[:4])[0]
            parts.append(payload[4:])
        else:
            parts.append(payload)

        if expected_total is not None and sum(len(p) for p in parts) >= expected_total:
            data = b"".join(parts)
            return data[:expected_total]

    data = b"".join(parts)
    if expected_total is None:
        raise ValueError("missing block 0")
    if len(data) < expected_total:
        raise ValueError(f"incomplete reassembly: got {len(data)}, expected {expected_total}")
    return data[:expected_total]
