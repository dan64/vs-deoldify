"""
------------------------------------------------------------------------------- 
Author: Jason Antic
Date: 2019-08-20
version: 
LastEditors: Dan64
LastEditTime: 2024-02-29
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
module that contains the main functions to control deoldify's colorization process.
This module has been modified to be compatible with Vapoursynth. All the calls to
ffmpeg, youtube, matplotlib were removed because non necessary in Vapoursynth.
All the interface functions has been changed to work with PIL images.
"""
from ..fastai.core import *
from ..fastai.vision import *
from .filters import IFilter, MasterFilter, ColorizerFilter
from .generators import gen_inference_deep, gen_inference_wide
from PIL import Image
import requests
from io import BytesIO
import base64
import cv2
import logging

class ModelImageVisualizer:
    def __init__(self, filter: IFilter, results_dir: str = None):
        self.filter = filter

    def _clean_mem(self):
        torch.cuda.empty_cache()
        # gc.collect()

    def _open_pil_image(self, path: Path) -> Image:
        return PIL.Image.open(path).convert('RGB')

    
    def get_transformed_pil_image(
        self,
        orig_image: Image,
        figsize: Tuple[int, int] = (20, 20),
        render_factor: int = None,
        post_process: bool = True,
    ) -> Image:
                
        result = self.get_transformed_orig_image(
            orig_image, render_factor, post_process=post_process)
        
        return result

    
    def get_transformed_orig_image(
        self, orig_image: Image, render_factor: int = None, post_process: bool = True,        
    ) -> Image:
        self._clean_mem()
        filtered_image = self.filter.filter(
            orig_image, orig_image, render_factor=render_factor,post_process=post_process
        )

        return filtered_image


    def _get_num_rows_columns(self, num_images: int, max_columns: int) -> Tuple[int, int]:
        columns = min(num_images, max_columns)
        rows = num_images // columns
        rows = rows if rows * columns == num_images else rows + 1
        return rows, columns


def get_artistic_video_colorizer(
    root_folder: Path = Path('./'),
    weights_name: str = 'ColorizeArtistic_gen',
    results_dir='result_images',
    render_factor: int = 35
) -> ModelImageVisualizer:
    learn = gen_inference_deep(root_folder=root_folder, weights_name=weights_name)
    filtr = MasterFilter([ColorizerFilter(learn=learn)], render_factor=render_factor)
    vis = ModelImageVisualizer(filtr, results_dir=results_dir)
    return vis


def get_stable_video_colorizer(
    root_folder: Path = Path('./'),
    weights_name: str = 'ColorizeVideo_gen',
    results_dir='result_images',
    render_factor: int = 21
) -> ModelImageVisualizer:
    learn = gen_inference_wide(root_folder=root_folder, weights_name=weights_name)
    filtr = MasterFilter([ColorizerFilter(learn=learn)], render_factor=render_factor)
    vis = ModelImageVisualizer(filtr, results_dir=results_dir)
    return vis


def get_image_colorizer(
    root_folder: Path = Path('./'), render_factor: int = 35, artistic: bool = True, isvideo: bool = False
) -> ModelImageVisualizer:
    if artistic:
        if isvideo:
            return get_artistic_video_colorizer(root_folder=root_folder, render_factor=render_factor)
        else:
            return get_artistic_image_colorizer(root_folder=root_folder, render_factor=render_factor)
    else:
        if isvideo:
            return get_stable_video_colorizer(root_folder=root_folder, render_factor=render_factor)
        else:
            return get_stable_image_colorizer(root_folder=root_folder, render_factor=render_factor)


def get_stable_image_colorizer(
    root_folder: Path = Path('./'),
    weights_name: str = 'ColorizeStable_gen',
    results_dir='result_images',
    render_factor: int = 35
) -> ModelImageVisualizer:
    learn = gen_inference_wide(root_folder=root_folder, weights_name=weights_name)
    filtr = MasterFilter([ColorizerFilter(learn=learn)], render_factor=render_factor)
    vis = ModelImageVisualizer(filtr, results_dir=results_dir)
    return vis


def get_artistic_image_colorizer(
    root_folder: Path = Path('./'),
    weights_name: str = 'ColorizeArtistic_gen',
    results_dir='result_images',
    render_factor: int = 35
) -> ModelImageVisualizer:
    learn = gen_inference_deep(root_folder=root_folder, weights_name=weights_name)
    filtr = MasterFilter([ColorizerFilter(learn=learn)], render_factor=render_factor)
    vis = ModelImageVisualizer(filtr, results_dir=results_dir)
    return vis
