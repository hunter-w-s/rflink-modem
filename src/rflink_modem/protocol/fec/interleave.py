from __future__ import annotations

def interleave_bytes(data: bytes, *, depth: int) -> bytes:
    """
    Block interleaver: write row-wise into depth rows, read column-wise.
    Deterministic and invertible (with same depth and original length).
    """
    if depth <= 1:
        return data
    n = len(data)
    if n == 0:
        return data

    cols = (n + depth - 1) // depth  # ceil(n / depth)
    # pad with a sentinel length using zero bytes; caller must pass original length to deinterleave
    padded_len = depth * cols
    pad = padded_len - n
    buf = data + b"\x00" * pad

    out = bytearray(padded_len)
    k = 0
    # read column-wise
    for c in range(cols):
        for r in range(depth):
            out[k] = buf[r * cols + c]
            k += 1

    return bytes(out)


def deinterleave_bytes(data: bytes, *, depth: int, original_len: int) -> bytes:
    """
    Inverse of interleave_bytes(). Requires original_len to remove padding unambiguously.
    """
    if depth <= 1:
        return data[:original_len]
    n = len(data)
    if n == 0:
        return data

    cols = (original_len + depth - 1) // depth
    padded_len = depth * cols
    if len(data) != padded_len:
        raise ValueError(f"unexpected interleaved length: got {len(data)}, expected {padded_len}")

    buf = bytearray(padded_len)
    k = 0
    # write column-wise back into row-wise storage
    for c in range(cols):
        for r in range(depth):
            buf[r * cols + c] = data[k]
            k += 1

    return bytes(buf[:original_len])
