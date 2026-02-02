import numpy as np
import pytest

from rflink_modem_v2.pipeline.stages.phy import stage as phy_stage


def _make_test_bits() -> np.ndarray:
    # deterministic, non-trivial pattern
    return np.array(([0, 1, 1, 0, 0, 1, 0, 1] * 64), dtype=np.uint8)


@pytest.mark.parametrize("module_name", phy_stage.available_modules())
def test_phy_stage_roundtrip_all_modules(module_name: str):
    """
    This test intentionally runs for *every* PHY module present in
    pipeline/stages/phy/modules.

    Contract tested:
      - module is selectable by stage Config(module=...)
      - stage.tx returns float32 PCM
      - stage.rx returns uint8 bits
      - roundtrip recovers original bits (for deterministic hard-decision PHY modules)
    """
    mod = phy_stage._import_phy_module(module_name)  # ok to use internal helper in tests

    # Require that module Config can be constructed with defaults.
    # This is an important project invariant: modules must be usable without special casing.
    try:
        module_cfg = mod.Config()
    except TypeError as e:
        pytest.fail(
            f"PHY module '{module_name}' Config() must be default-constructible. "
            f"Fix module defaults or provide a minimal default config. Error: {e}"
        )

    cfg = phy_stage.Config(module=module_name, module_cfg=module_cfg, use_cached_demod=True)

    bits = _make_test_bits()
    pcm = phy_stage.tx(bits, cfg=cfg)

    assert isinstance(pcm, np.ndarray)
    assert pcm.dtype == np.float32
    assert pcm.ndim == 1
    assert len(pcm) > 0

    out = phy_stage.rx(pcm, cfg=cfg)

    assert isinstance(out, np.ndarray)
    assert out.dtype == np.uint8
    assert out.shape == bits.shape
    assert np.array_equal(out, bits), f"roundtrip failed for module={module_name}"
