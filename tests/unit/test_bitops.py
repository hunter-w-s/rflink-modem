from pathlib import Path
from rflink_modem.utils.bitops import bytes_to_bits, bits_to_bytes

def test_png_roundtrip():
    src = Path("tests/assets/cwg.png")
    out_png = Path("tests/output/cwg_roundtrip.png")
    out_bits = Path("tests/output/cwg_bits.txt")

    data = src.read_bytes()

    bits = bytes_to_bits(data)
    recovered = bits_to_bytes(bits)

    # Debug artifacts (early-stage sanity checks)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    out_png.write_bytes(recovered)

    # Save bits as text: one continuous stream
    out_bits.write_text("".join(str(b) for b in bits))

    assert recovered == data
