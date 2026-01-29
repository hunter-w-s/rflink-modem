from pathlib import Path

from rflink_modem.modem.audio.afsk_tx import AFSKTxConfig, bits_to_wav


def read_bits_txt(path: str) -> list[int]:
    s = Path(path).read_text().strip()
    return [1 if ch == "1" else 0 for ch in s]


if __name__ == "__main__":
    bits = read_bits_txt("tests/output/reference/cwg_bits.txt")

    cfg = AFSKTxConfig(
        sample_rate=48000,
        symbol_rate=1200,
        mark_hz=2400.0,
        space_hz=1200.0,
        amplitude=0.8,
        lead_silence_s=0.25,
        trail_silence_s=0.25,
    )
    rf_intent = {
    "center_freq_hz": 2_437_000_000,   # 2.437 GHz (Wi-Fi Ch 6 center, as an example)
    "occupied_bw_hz": 20_000,          # “feasible” narrowband channel (e.g., 20 kHz)
    "modulation": "2-FSK",
    "symbol_rate": cfg.symbol_rate,
    "freq_deviation_hz": (cfg.mark_hz - cfg.space_hz) / 2,  # 600 Hz deviation (in baseband terms)
    }

    out = "tests/output/reference/cwg_afsk_tx.wav"
    bits_to_wav(bits, out, cfg)
    print(f"Wrote {out}")
