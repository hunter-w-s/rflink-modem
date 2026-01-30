# rflink_modem/modem/audio/api.py

def afsk_tx_bits(bits, cfg):
    return bits_to_pcm(bits, cfg)

def afsk_rx_bits(pcm, cfg):
    return demod_pcm_to_bits(pcm, cfg)
