"""
Convert the Open-CD SNUNet checkpoint into backend/models/snunet_levir.pt.

Downloads (if missing) the LEVIR-CD SNUNet weights from the Open-CD model
zoo on HuggingFace and strips the mmengine wrapper so the checkpoint can be
loaded without mmengine installed:

    source:  https://huggingface.co/likyoo/Open-CD_Model_Zoo/resolve/main/LEVIR-CD/snunet_c16_256x256_40k_levircd.pth
    output:  backend/models/snunet_levir.pt

The output contains a plain dict {"state_dict": {...}} whose keys match
src/snunet.py exactly (verified with strict=True before saving).

Usage (from backend/):
    ..\\venv\\Scripts\\python.exe scripts/convert_snunet.py
"""

import sys
import logging
import tempfile
import urllib.request
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("convert_snunet")

MODEL_URL = ("https://huggingface.co/likyoo/Open-CD_Model_Zoo/resolve/main/"
             "LEVIR-CD/snunet_c16_256x256_40k_levircd.pth")
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
OUTPUT_PATH = MODELS_DIR / "snunet_levir.pt"


def download(url, dest):
    if dest.exists():
        logger.info("Using cached %s", dest)
        return dest
    logger.info("Downloading %s", url)
    tmp = Path(tempfile.gettempdir()) / "snunet_c16.pth"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, tmp)
    logger.info("Downloaded %d bytes", tmp.stat().st_size)
    return tmp


def main():
    tmp = download(MODEL_URL, OUTPUT_PATH)

    import torch
    # mmengine's Checkpoint is a dict subclass; we only need its items.
    # mmengine must be importable (it was installed in the dev venv).
    import mmengine  # noqa: F401  (registers the checkpoint class)

    logger.info("Loading source checkpoint (requires mmengine)")
    ckpt = torch.load(str(tmp), map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        state_dict = ckpt
    else:
        raise RuntimeError(f"Unexpected checkpoint type: {type(ckpt)}")

    state_dict = {k: v for k, v in state_dict.items() if k.startswith("backbone.") or k.startswith("decode_head.")}

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.snunet import SNUNet

    model = SNUNet(in_ch=3, base_channel=16)
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    assert not missing and not unexpected, f"Key mismatch: missing={missing} unexpected={unexpected}"
    logger.info("All %d tensors matched (strict=True)", len(state_dict))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": state_dict}, str(OUTPUT_PATH))
    logger.info("Saved %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()