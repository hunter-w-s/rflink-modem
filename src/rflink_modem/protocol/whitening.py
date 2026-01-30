# src/rflink_modem/protocol/whitening.py
from __future__ import annotations


def whiten(data: bytes, *, seed: int = 0xACE1) -> bytes:
    """
    XOR whitening using a 16-bit LFSR. Symmetric: whiten(whiten(x)) == x.

    - seed must be non-zero 16-bit value.
    - Generates one pseudo-random byte per input byte and XORs it.

    This is NOT crypto. It's for spectral/run-length properties.
    """
    if not (0 <= seed <= 0xFFFF) or seed == 0:
        raise ValueError("seed must be a non-zero 16-bit value")

    lfsr = seed & 0xFFFF
    out = bytearray(len(data))

    for i, b in enumerate(data):
        # Generate 8 bits from the LFSR to form one byte
        prn = 0
        for _ in range(8):
            # taps: x^16 + x^12 + x^5 + 1 => 0x1021 (CCITT) in non-reflected form
            # We'll implement as: new_bit = (bit15 ^ bit11 ^ bit4 ^ bit0)
            bit15 = (lfsr >> 15) & 1
            bit11 = (lfsr >> 11) & 1
            bit4 = (lfsr >> 4) & 1
            bit0 = lfsr & 1
            new_bit = bit15 ^ bit11 ^ bit4 ^ bit0

            lfsr = ((lfsr << 1) & 0xFFFF) | new_bit
            prn = ((prn << 1) | (lfsr & 1)) & 0xFF

        out[i] = (b ^ prn) & 0xFF

    return bytes(out)
