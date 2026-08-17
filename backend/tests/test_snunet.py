"""Tests for the SNUNet change detector (Open-CD port + converted weights)."""

from pathlib import Path

import torch

from src.snunet import SNUNet


def _model():
    model = SNUNet(in_ch=3, base_channel=16)
    model.eval()
    return model


def test_forward_shape():
    model = _model()
    with torch.no_grad():
        out = model(torch.randn(1, 3, 32, 32), torch.randn(1, 3, 32, 32))
    assert tuple(out.shape) == (1, 2, 32, 32)


def test_forward_larger_tile():
    model = _model()
    with torch.no_grad():
        out = model(torch.randn(1, 3, 512, 512), torch.randn(1, 3, 512, 512))
    assert tuple(out.shape) == (1, 2, 512, 512)


def test_checkpoint_loads_strictly():
    from pathlib import Path

    from src.config import BASE_DIR

    ckpt_path = Path(BASE_DIR) / "models" / "snunet_levir.pt"
    if not ckpt_path.exists():
        import pytest
        pytest.skip("snunet_levir.pt not present — run scripts/convert_snunet.py")

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    missing, unexpected = _model().load_state_dict(state_dict, strict=True)
    assert not missing and not unexpected
    assert len(state_dict) > 100  # full pretrained weights, not a stub


def test_detect_real_levir_change(tmp_path):
    """The pretrained model must fire on a real LEVIR building change and
    stay quiet on a purely random pair (which is far out of distribution)."""
    import cv2
    import numpy as np
    import pytest

    from src.models import ModelManager

    levir_a = "C:/Users/Admin/AppData/Local/Temp/opencode/levir_a.png"
    levir_b = "C:/Users/Admin/AppData/Local/Temp/opencode/levir_b.png"
    if not Path(levir_a).exists() or not Path(levir_b).exists():
        pytest.skip("LEVIR sample pair not available")

    pa, pb = tmp_path / "a.png", tmp_path / "b.png"
    cv2.imwrite(str(pa), cv2.imread(levir_a))
    cv2.imwrite(str(pb), cv2.imread(levir_b))

    manager = ModelManager()
    assert manager.models.get("snunet_available") is True

    prob = manager._prob_with_snunet(str(pa), str(pb))
    assert prob.shape == (256, 256)
    assert float(prob.max()) > 0.5, "SNUNet did not activate on real LEVIR change"

    rng = np.random.default_rng(0)
    na, nb = tmp_path / "noise_a.png", tmp_path / "noise_b.png"
    cv2.imwrite(str(na), rng.integers(0, 255, (256, 256, 3), dtype=np.uint8))
    cv2.imwrite(str(nb), rng.integers(0, 255, (256, 256, 3), dtype=np.uint8))
    prob_noise = manager._prob_with_snunet(str(na), str(nb))
    assert float(prob_noise.max()) < 0.1, "SNUNet should not fire on random noise"


def test_changeformer_native_256_protocol(tmp_path):
    """ChangeFormer V6 only works at exactly 256x256 (the repo's demo
    resizes to 256 before inference).  It must fire on a real LEVIR change
    when fed via the native-256 protocol, and stay quiet on random noise."""
    import cv2
    import numpy as np
    import pytest

    from src.models import ModelManager

    levir_a = "C:/Users/Admin/AppData/Local/Temp/opencode/levir_a.png"
    levir_b = "C:/Users/Admin/AppData/Local/Temp/opencode/levir_b.png"
    if not Path(levir_a).exists() or not Path(levir_b).exists():
        pytest.skip("LEVIR sample pair not available")

    pa, pb = tmp_path / "a.png", tmp_path / "b.png"
    cv2.imwrite(str(pa), cv2.imread(levir_a))
    cv2.imwrite(str(pb), cv2.imread(levir_b))

    manager = ModelManager()
    assert manager.models.get("changeformer_available") is True

    prob = manager._prob_with_changeformer(str(pa), str(pb))
    assert prob.shape == (256, 256)
    assert float(prob.max()) > 0.3, "ChangeFormer did not activate on real LEVIR change"

    rng = np.random.default_rng(0)
    na, nb = tmp_path / "noise_a.png", tmp_path / "noise_b.png"
    cv2.imwrite(str(na), rng.integers(0, 255, (256, 256, 3), dtype=np.uint8))
    cv2.imwrite(str(nb), rng.integers(0, 255, (256, 256, 3), dtype=np.uint8))
    prob_noise = manager._prob_with_changeformer(str(na), str(nb))
    assert float(prob_noise.max()) < 0.3, "ChangeFormer should not fire on random noise"