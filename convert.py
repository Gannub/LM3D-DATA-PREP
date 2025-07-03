import os
import tyro
from dataclasses import dataclass
from converter import convert
from typing import Literal, Annotated

@dataclass
class Config:
    """
    Convert Lumio3D output to a dataset that is
    compatible with VHAP.
    
    INPUT File structure:

    --- input_dir/
        --- cameras/ (directly obtained from Lumio3D)
            --- cam_0.json
            --- cam_1.json
            ...
        --- data/ (directly optained from Lumio3D)
            --- cam_0/
                --- diff.exr
                --- flash_0.exr
                --- flash_1.exr
                --- diff_normal.exr
                --- mask.png
            --- cam_1/
                ...
    """

    input_dir: Annotated[str, tyro.conf.arg(aliases=['-i'])]
    """Input directory containing Lumio3D output."""
    output_dir: Annotated[str, tyro.conf.arg(aliases=['-o'])]
    """Output directory for the converted dataset."""
    width : int = 435
    """Width of the output images."""
    height: int = 574
    """Height of the output images."""
    image_type: Annotated[Literal['diff', 'flash_0', 'flash_1', 'diff_normal'], tyro.conf.arg(aliases=['-itype'])] = 'flash_1'
    """Type of input images to convert."""


if __name__ == "__main__":
    os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
    config = tyro.cli(Config)
    convert(
        input_dir=config.input_dir,
        out_dir=config.output_dir,
        width=config.width,
        height=config.height,
        image_type=config.image_type
    )

