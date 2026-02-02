from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from rflink_modem_v2.pipeline.config import PipelineConfig

from rflink_modem_v2.pipeline.stages.sync import stage as sync_stage
from rflink_modem_v2.pipeline.stages.crypto import stage as crypto_stage
from rflink_modem_v2.pipeline.stages.framing import stage as framing_stage
from rflink_modem_v2.pipeline.stages.whiten import stage as whiten_stage
from rflink_modem_v2.pipeline.stages.interleave import stage as interleave_stage
from rflink_modem_v2.pipeline.stages.blocking import stage as blocking_stage
from rflink_modem_v2.pipeline.stages.byte_fec import stage as byte_fec_stage
from rflink_modem_v2.pipeline.stages.bit_fec import stage as bit_fec_stage
from rflink_modem_v2.pipeline.stages.phy import stage as phy_stage


def _pack_bits_msb_fullbytes(bits: np.ndarray) -> tuple[bytes, np.ndarray]:
    """
    Pack bits into bytes MSB-first, but ONLY for full bytes.
    Returns (packed_bytes, remainder_bits).
    """
    bits = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if bits.size == 0:
        return b"", np.zeros((0,), dtype=np.uint8)

    n_full = (bits.size // 8) * 8
    if n_full == 0:
        return b"", bits

    full = bits[:n_full]
    rem = bits[n_full:]

    out = bytearray(n_full // 8)
    j = 0
    for i in range(0, n_full, 8):
        v = 0
        for k in range(8):
            v = (v << 1) | int(full[i + k] & 1)
        out[j] = v
        j += 1
    return bytes(out), rem


def _unpack_bits_msb(b: bytes) -> np.ndarray:
    if not b:
        return np.zeros((0,), dtype=np.uint8)
    arr = np.frombuffer(b, dtype=np.uint8)
    out = np.empty(arr.size * 8, dtype=np.uint8)
    o = 0
    for x in arr:
        for k in range(7, -1, -1):
            out[o] = (x >> k) & 1
            o += 1
    return out


def _rs_default_cfg_if_none(byte_fec_stage_cfg):
    """
    Return a concrete rs255 module_cfg (default-constructed if None).
    Only used when module == "rs255".
    """
    if byte_fec_stage_cfg.module != "rs255":
        return byte_fec_stage_cfg.module_cfg
    if byte_fec_stage_cfg.module_cfg is not None:
        return byte_fec_stage_cfg.module_cfg

    # Instantiate module default so we match real defaults (e.g. nsym=32).
    from rflink_modem_v2.pipeline.stages.byte_fec.modules.rs255 import Config as RS255Cfg
    return RS255Cfg()


def _get_rs_k(byte_fec_stage_cfg) -> int:
    rs_cfg = _rs_default_cfg_if_none(byte_fec_stage_cfg)
    nsym = getattr(rs_cfg, "nsym", None)
    if not isinstance(nsym, int):
        raise ValueError("rs255 Config must have int nsym")
    return 255 - nsym


def _pick_block_depth_for_k(k: int, desired_max: int = 8) -> int:
    # pick largest divisor <= desired_max; if k is prime relative to <=8, fallback to 1
    for d in range(desired_max, 1, -1):
        if k % d == 0:
            return d
    return 1

def _pack_bits_msb_fullbytes(bits: np.ndarray) -> bytes:
    bits = np.asarray(bits, dtype=np.uint8).reshape(-1)
    n_full = (bits.size // 8) * 8
    if n_full == 0:
        return b""
    bits = bits[:n_full]

    out = bytearray(n_full // 8)
    j = 0
    for i in range(0, n_full, 8):
        v = 0
        for k in range(8):
            v = (v << 1) | int(bits[i + k] & 1)
        out[j] = v
        j += 1
    return bytes(out)



@dataclass
class Pipeline:
    cfg: PipelineConfig

    # Buffers / remainders
    _pcm_buf: np.ndarray = np.zeros((0,), dtype=np.float32)
    _bit_rem: np.ndarray = np.zeros((0,), dtype=np.uint8)
    _rs_rem: bytes = b""              # <-- ADD
    _sync_rem: bytes = b""
    _frame_rem: bytes = b""

    def reset(self) -> None:
        self._pcm_buf = np.zeros((0,), dtype=np.float32)
        self._bit_rem = np.zeros((0,), dtype=np.uint8)
        self._rs_rem = b""            # <-- ADD
        self._sync_rem = b""
        self._frame_rem = b""


    def _effective_interleave_cfg(self):
        """
        Pipeline policy glue:
        For interleave.block, enforce a depth compatible with RS k.
        If a user-provided depth is incompatible, we override it to the nearest valid depth.
        """
        if self.cfg.interleave.module != "block":
            return self.cfg.interleave

        from rflink_modem_v2.pipeline.stages.interleave.stage import Config as InterleaveStageConfig
        from rflink_modem_v2.pipeline.stages.interleave.modules.block import Config as BlockCfg

        # Determine RS k using the REAL rs255 defaults if module_cfg is None
        k = _get_rs_k(self.cfg.byte_fec)

        # Start from configured depth if provided, else assume "desired_max=8"
        desired = 8
        if self.cfg.interleave.module_cfg is not None:
            desired = int(getattr(self.cfg.interleave.module_cfg, "depth", 8))

        depth = _pick_block_depth_for_k(k, desired_max=desired)

        # Always return a block cfg that is valid for k
        return InterleaveStageConfig(module="block", module_cfg=BlockCfg(depth=depth, pad=0x00))


    # -------------------------
    # TX: payload -> PCM
    # -------------------------

    def tx(self, payload: bytes) -> np.ndarray:
        if not isinstance(payload, (bytes, bytearray)):
            raise TypeError("Pipeline.tx: payload must be bytes-like")

        x = bytes(payload)

        # Encrypt/authenticate payload first
        x = crypto_stage.tx(x, cfg=self.cfg.crypto)

        # Frame then sync
        x = framing_stage.tx(x, cfg=self.cfg.framing)
        x = sync_stage.tx(x, cfg=self.cfg.sync)

        # Channel transforms
        x = whiten_stage.tx(x, cfg=self.cfg.whiten)

        # blocking BEFORE interleave
        x = blocking_stage.tx(x, cfg=self.cfg.blocking)

        il_cfg = self._effective_interleave_cfg()
        x = interleave_stage.tx(x, cfg=il_cfg)

        x = byte_fec_stage.tx(x, cfg=self.cfg.byte_fec)
        x = bit_fec_stage.tx(x, cfg=self.cfg.bit_fec)  # packed-bits bytes

        bits = _unpack_bits_msb(x)
        pcm = phy_stage.tx(bits, cfg=self.cfg.phy)
        return np.asarray(pcm, dtype=np.float32)

    # -------------------------
    # RX: PCM -> payloads
    # -------------------------

    def rx(self, pcm: np.ndarray) -> List[bytes]:
        pcm = np.asarray(pcm, dtype=np.float32).reshape(-1)
        if pcm.size == 0:
            return []

        # Accumulate burst PCM
        if self._pcm_buf.size == 0:
            self._pcm_buf = pcm
        else:
            self._pcm_buf = np.concatenate([self._pcm_buf, pcm], axis=0)

        try:
            bits = phy_stage.rx(self._pcm_buf, cfg=self.cfg.phy)
            bits = np.asarray(bits, dtype=np.uint8).reshape(-1)

            # Preserve non-byte-aligned demod output by buffering remainder bits
            packed = _pack_bits_msb_fullbytes(bits)
            if not packed:
                return []


            # Reverse channel transforms
            # Reverse channel transforms
            x = bit_fec_stage.rx(packed, cfg=self.cfg.bit_fec)

            # ---- RS boundary buffering: only decode full 255-byte codewords ----
            x = byte_fec_stage.rx(x, cfg=self.cfg.byte_fec)
            
            il_cfg = self._effective_interleave_cfg()
            x = interleave_stage.rx(x, cfg=il_cfg)

            x = blocking_stage.rx(x, cfg=self.cfg.blocking)
            x = whiten_stage.rx(x, cfg=self.cfg.whiten)

            # Sync (optional) + framing
            byte_stream = self._sync_rem + x
            aligned, sync_rem = sync_stage.rx(byte_stream, cfg=self.cfg.sync)

            if aligned:
                stream_for_framing = aligned
                self._sync_rem = sync_rem
            else:
                # fallback to framing’s own magic resync
                stream_for_framing = byte_stream
                self._sync_rem = b""

            payloads, frame_rem = framing_stage.rx(self._frame_rem + stream_for_framing, cfg=self.cfg.framing)
            self._frame_rem = frame_rem

            out: List[bytes] = []
            for p in payloads:
                try:
                    out.append(crypto_stage.rx(p, cfg=self.cfg.crypto))
                except ValueError:
                    continue

            if out:
                # one-burst assumption
                self._pcm_buf = np.zeros((0,), dtype=np.float32)
                self._bit_rem = np.zeros((0,), dtype=np.uint8)
                self._sync_rem = b""
                self._frame_rem = b""

            return out

        except ValueError:
            # Not enough accumulated data yet (chunked RX) OR partial decode artifact
            return[]
