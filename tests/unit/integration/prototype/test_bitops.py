from pathlib import Path

from rflink_modem.utils.bitops import bytes_to_bits, bits_to_bytes
from tests.conftest import sample_assets_dir, outputs_unrev_test_dir


def test_png_roundtrip(request):
    src = sample_assets_dir() / "cwg.png"

    # generated artifacts go here (not committed) + per-test folder
    out_dir = outputs_unrev_test_dir(request)

    out_png = out_dir / "cwg_roundtrip.png"
    out_bits = out_dir / "cwg_bits.txt"

    data = src.read_bytes()

    bits = bytes_to_bits(data)
    recovered = bits_to_bytes(bits)

    out_png.write_bytes(recovered)
    out_bits.write_text("".join(str(b) for b in bits), encoding="utf-8")

    assert recovered == data
