"""
------------------------------------------------------------------------------- 
Author: Jason Antic
Date: 2019-08-20
version: 
LastEditors: Dan64
LastEditTime: 2024-04-08
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

    def __init__(self, filter: IFilter):
        self.filter = filter

    def _clean_mem(self):
        torch.cuda.empty_cache()
        # gc.collect()

    def get_transformed_image(self, orig_image: Image, render_factor: int = None, post_process: bool = True) -> Image:
                
        self._clean_mem()
        filtered_image = self.filter.filter(orig_image, orig_image, render_factor=render_factor, post_process=post_process)

        return filtered_image

class ModelImageInitializer:
    _instance = None
    _initialized = False
    _artistic_video = None
    _artistic_image = None
    _stable_video = None
    _stable_image = None
    _render_factor = 0

    package_dir='./'

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, package_dir: str = None):

        if not self._initialized:
            self.package_dir = package_dir
            self._initialized = True

    def get_image_colorizer(self, render_factor: int = 34, artistic: bool = True, isvideo: bool = False) -> ModelImageVisualizer:

        root_folder = Path(self.package_dir)

        if artistic:
            if isvideo:
                return self._get_artistic_video_colorizer(render_factor=render_factor)
            else:
                return self._get_artistic_image_colorizer(render_factor=render_factor)
        else:
            if isvideo:
                return self._get_stable_video_colorizer(render_factor=render_factor)
            else:
                return self._get_stable_image_colorizer(render_factor=render_factor)

    def _get_stable_image_colorizer(self, weights_name: str = 'ColorizeStable_gen',
        render_factor: int = 34) -> ModelImageVisualizer:

        root_folder = Path(self.package_dir)

        if not (self._stable_image is None) and render_factor==self._render_factor:
            return self._stable_image

        learn = gen_inference_wide(root_folder=root_folder, weights_name=weights_name)
        filtr = MasterFilter([ColorizerFilter(learn=learn)], render_factor=render_factor)

        self._stable_image = ModelImageVisualizer(filtr)
        self._render_factor = render_factor

        return self._stable_image

    def _get_artistic_image_colorizer(self, weights_name: str = 'ColorizeArtistic_gen',
        render_factor: int = 34) -> ModelImageVisualizer:

        root_folder = Path(self.package_dir)

        if not (self._artistic_image is None) and render_factor==self._render_factor:
            return self._artistic_image

        learn = gen_inference_deep(root_folder=root_folder, weights_name=weights_name)
        filtr = MasterFilter([ColorizerFilter(learn=learn)], render_factor=render_factor)

        self._artistic_image = ModelImageVisualizer(filtr)
        self._render_factor = render_factor

        return self._artistic_image

    def _get_artistic_video_colorizer(self, weights_name: str = 'ColorizeArtistic_gen',
        render_factor: int = 35) -> ModelImageVisualizer:

        root_folder = Path(self.package_dir)

        if not (self._artistic_video is None) and render_factor==self._render_factor:
            return self._artistic_video

        learn = gen_inference_deep(root_folder=root_folder, weights_name=weights_name)
        filtr = MasterFilter([ColorizerFilter(learn=learn)], render_factor=render_factor)

        self._artistic_video = ModelImageVisualizer(filtr)
        self._render_factor = render_factor

        return self._artistic_video


    def _get_stable_video_colorizer(self, weights_name: str = 'ColorizeVideo_gen',
        render_factor: int = 21) -> ModelImageVisualizer:

        root_folder = Path(self.package_dir)

        if not (self._stable_video is None) and render_factor==self._render_factor:
            return self._stable_video

        learn = gen_inference_wide(root_folder=root_folder, weights_name=weights_name)
        filtr = MasterFilter([ColorizerFilter(learn=learn)], render_factor=render_factor)

        self._stable_video = ModelImageVisualizer(filtr)
        self._render_factor = render_factor

        return self._stable_video
