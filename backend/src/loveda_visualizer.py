import numpy as np

from PIL import Image

from src.loveda_colors import COLORS


def colorize(mask):

    """
    Convert LoveDA semantic mask into RGB visualization.

    Classes:
    0 Ignore
    1 Background
    2 Building
    3 Road
    4 Water
    5 Barren
    6 Forest
    7 Agricultural
    """

    if mask is None:
        raise ValueError(
            "Semantic mask is empty"
        )

    if len(mask.shape) != 2:
        raise ValueError(
            "Semantic mask must be 2D"
        )

    h, w = mask.shape

    # Default unknown class color

    output = np.full(
        (h, w, 3),
        fill_value=(
            128,
            128,
            128
        ),
        dtype=np.uint8
    )

    for cls, color in COLORS.items():

        output[
            mask == cls
        ] = color

    return Image.fromarray(
        output
    )