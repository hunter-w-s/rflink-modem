import pytest

from rflink_modem_v2.pipeline.stages.framing.modules.length import Config, tx, rx


def test_length_framing_roundtrip():
    cfg = Config(
        magic=b"\xA5\xC3",
        version=1,
        flags=0,
        max_payload=4096,
        seq=123,
    )

    payload = b"hello framing\x00\x01\x02"
    framed = tx(payload, cfg=cfg)
    out = rx(framed, cfg=cfg)

    assert out == payload


def test_length_framing_rejects_payload_corruption():
    cfg = Config(max_payload=4096, seq=1)
    payload = b"abcdefg"
    framed = bytearray(tx(payload, cfg=cfg))

    # Flip one byte in payload region (after 12-byte header)
    # Header: 10 bytes + 2 bytes hdr_crc = 12 bytes
    framed[12] ^= 0x01

    with pytest.raises(ValueError, match="CRC32"):
        rx(bytes(framed), cfg=cfg)


def test_length_framing_rejects_header_corruption():
    cfg = Config(max_payload=4096, seq=2)
    payload = b"abcdefg"
    framed = bytearray(tx(payload, cfg=cfg))

    # Corrupt the header (e.g., flags byte is at offset:
    # MAGIC(2) + VER(1) => FLAGS at index 3
    framed[3] ^= 0x01

    # Header CRC16 should fail
    with pytest.raises(ValueError, match="header CRC16"):
        rx(bytes(framed), cfg=cfg)


def test_length_framing_rejects_bad_magic():
    cfg = Config(magic=b"\xA5\xC3", seq=3)
    payload = b"ok"
    framed = tx(payload, cfg=cfg)

    wrong_cfg = Config(magic=b"\xDE\xAD", seq=3)

    with pytest.raises(ValueError, match="bad magic"):
        rx(framed, cfg=wrong_cfg)
