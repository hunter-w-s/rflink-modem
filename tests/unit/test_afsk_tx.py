import numpy as np

from rflink_modem.modem.audio.afsk_tx import AFSKTxConfig, bits_to_pcm, samples_per_symbol


def test_samples_per_symbol_exact():
    cfg = AFSKTxConfig(sample_rate=48000, symbol_rate=1200)
    assert samples_per_symbol(cfg) == 40


def test_bits_to_pcm_length_and_silence():
    cfg = AFSKTxConfig(
        sample_rate=48000,
        symbol_rate=1200,
        mark_hz=2400.0,
        space_hz=1200.0,
        amplitude=0.8,
        lead_silence_s=0.25,
        trail_silence_s=0.25,
    )

    bits = [0, 1, 1, 0, 0, 1]
    pcm = bits_to_pcm(bits, cfg)

    sps = samples_per_symbol(cfg)
    lead_n = int(round(cfg.lead_silence_s * cfg.sample_rate))
    trail_n = int(round(cfg.trail_silence_s * cfg.sample_rate))

    expected = lead_n + len(bits) * sps + trail_n
    assert pcm.dtype == np.float32
    assert pcm.shape == (expected,)

    # Lead/trail should be exactly zeros
    assert np.all(pcm[:lead_n] == 0.0)
    assert np.all(pcm[-trail_n:] == 0.0)

    # Signal portion should be non-zero (most of the time)
    body = pcm[lead_n : lead_n + len(bits) * sps]
    assert np.max(np.abs(body)) > 0.1


def test_bits_to_pcm_amplitude_bound():
    cfg = AFSKTxConfig(sample_rate=48000, symbol_rate=1200, amplitude=0.8)
    bits = [0, 1] * 100
    pcm = bits_to_pcm(bits, cfg)

    # Should stay within [-amplitude, amplitude] (sin oscillator scaled)
    assert float(np.max(pcm)) <= cfg.amplitude + 1e-3
    assert float(np.min(pcm)) >= -cfg.amplitude - 1e-3
