"""Small compatibility launcher for the external YOLOv5 training script."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Expected path to YOLOv5 train.py")
    train_script = Path(sys.argv[1]).expanduser().resolve()
    if not train_script.is_file():
        raise SystemExit(f"YOLOv5 training entry does not exist: {train_script}")

    # YOLOv5 7.0 calls FreeTypeFont.getsize(), removed by Pillow 10. Keep the
    # trainer checkout untouched while restoring that small compatibility API.
    try:
        from PIL import ImageFont
        if not hasattr(ImageFont.FreeTypeFont, "getsize"):
            def getsize(font, text, *args, **kwargs):
                left, top, right, bottom = font.getbbox(text)
                return right - left, bottom - top
            ImageFont.FreeTypeFont.getsize = getsize
    except ImportError:
        pass

    # PyTorch 2.6+ changed the default to weights_only=True, while YOLOv5
    # checkpoints serialize model objects and optimizer state for resume.
    try:
        import torch
        original_torch_load = torch.load

        def trusted_checkpoint_load(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return original_torch_load(*args, **kwargs)

        torch.load = trusted_checkpoint_load
    except ImportError:
        pass

    sys.argv = [str(train_script), *sys.argv[2:]]
    runpy.run_path(str(train_script), run_name="__main__")


if __name__ == "__main__":
    main()
