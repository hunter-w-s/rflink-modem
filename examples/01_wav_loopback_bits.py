from pathlib import Path

from rflink_modem.modem.audio.afsk_tx import AFSKTxConfig, bits_to_wav
from rflink_modem.modem.audio.afsk_rx import demod_wav_to_bits


def read_bits_txt(path: str) -> list[int]:
    s = Path(path).read_text().strip()
    return [1 if ch == "1" else 0 for ch in s]


def write_bits_txt(path: str, bits: list[int]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join("1" if b else "0" for b in bits))


if __name__ == "__main__":
    cfg = AFSKTxConfig(
        sample_rate=48000,
        symbol_rate=1200,
        mark_hz=2400.0,
        space_hz=1200.0,
        amplitude=0.8,
        lead_silence_s=0.25,
        trail_silence_s=0.25,
    )

    bits_in = read_bits_txt("tests/output/reference/cwg_bits.txt")

    wav_out = "tests/output/reference/cwg_afsk_tx.wav"
    bits_to_wav(bits_in, wav_out, cfg)

    bits_out = demod_wav_to_bits(wav_out, cfg).tolist()
    write_bits_txt("tests/output/reference/cwg_bits_demod.txt", bits_out)

    # Hard correctness check
    assert bits_out == bits_in, "Demod mismatch: bits_out != bits_in"
    print("Loopback OK: demod bits match exactly.")
