from pathlib import Path
from rflink_modem.utils.bitops import bytes_to_bits, bits_to_bytes

def test_png_roundtrip():
    src = Path("tests/assets/cwg.png")
    data = src.read_bytes()

    bits = bytes_to_bits(data)
    recovered = bits_to_bytes(bits)

    assert recovered == data
