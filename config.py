# create a dataclass config

from dataclasses import dataclass
from typing import List


@dataclass
class Config:
    seed_values: List[int] = [8, 16, 50, 80, 128]
    height = 512
    width = 512
    num_inference_steps = 10
    guidance_scale = 7.5

    style_files = [
        "./sd-concept-library/arcane_style.bin",
        "./sd-concept-library/birb_style.bin",
        "./sd-concept-library/hitokomoru_style.bin",
        "./sd-concept-library/line_art_style.bin",
        "./sd-concept-library/sakimi_style.bin",
    ]

    num_styles = len(style_files)

    batch_size = 1
    num_inference_steps = 30
