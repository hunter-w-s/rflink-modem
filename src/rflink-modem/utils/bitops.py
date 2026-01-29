from pathlib import Path
from typing import List

def bytes_to_bits(data: bytes, msb_first: bool = True) -> List[int]:
    bits = []
    if msb_first:
        for b in data:
            for i in range(7, -1, -1):
                bits.append((b >> i) & 1)
    else:
        for b in data:
            for i in range(8):
                bits.append((b >> i) & 1)
    return bits

def bits_to_bytes(bits: List[int], msb_first: bool = True) -> bytes:
    if len(bits) % 8 != 0:
        raise ValueError(f"Bit length must be multiple of 8, got {len(bits)}")
    out = bytearray(len(bits) // 8)
    if msb_first:
        for bi in range(0, len(bits), 8):
            v = 0
            for i in range(8):
                v = (v << 1) | (bits[bi + i] & 1)
            out[bi // 8] = v
    else:
        for bi in range(0, len(bits), 8):
            v = 0
            for i in range(8):
                v |= (bits[bi + i] & 1) << i
            out[bi // 8] = v
    return bytes(out)

def png_file_to_bits(png_path: str | Path, msb_first: bool = True) -> List[int]:
    data = Path(png_path).read_bytes()
    return bytes_to_bits(data, msb_first=msb_first)

def bits_to_png_file(bits: List[int], out_path: str | Path, msb_first: bool = True) -> None:
    data = bits_to_bytes(bits, msb_first=msb_first)
    Path(out_path).write_bytes(data)
