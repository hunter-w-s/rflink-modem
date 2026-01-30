# tests/unit/test_crc.py
from __future__ import annotations

from rflink_modem.protocol.crc import crc16_ccitt_false, crc32_ieee, crc16, crc32



def test_crc16_ccitt_false_known_vector():
    assert crc16_ccitt_false(b"123456789") == 0x29B1


def test_crc32_ieee_known_vector():
    assert crc32_ieee(b"123456789") == 0xCBF43926


def test_crc16_alias_matches():
    assert crc16(b"123456789") == 0x29B1


def test_crc32_alias_matches():
    assert crc32(b"123456789") == 0xCBF43926


def test_crc_empty_vectors():
    # With init=0xFFFF, CCITT-FALSE of empty is 0xFFFF
    assert crc16_ccitt_false(b"") == 0xFFFF
    # With init=0xFFFFFFFF and xorout=0xFFFFFFFF, CRC32 of empty is 0x00000000
    assert crc32_ieee(b"") == 0x00000000
