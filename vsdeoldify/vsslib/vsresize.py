"""
-------------------------------------------------------------------------------
Author: Dan64
Date: 2024-05-15
version:
LastEditors: Dan64
LastEditTime: 2024-10-17
-------------------------------------------------------------------------------
Description:
-------------------------------------------------------------------------------
Vapoursynth Smart Resize class.
"""

import vapoursynth as vs
import math
import numpy as np
import cv2
from PIL import Image


"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
Class for resize clips in 16/9 aspect ratio, borders are added if needed 
"""


class SmartResizeColorizer:
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, clip_size: list = [432, 768]):

        if not self._initialized:
            self.target_width = clip_size[1]
            self.target_height = clip_size[0]
            self.ratio_target = round(self.target_width / self.target_height, 2)
            self._initialized = True

    def get_resized_clip(self, clip: vs.VideoNode) -> vs.VideoNode:
        self.clip_w = clip.width
        self.clip_h = clip.height
        self.ratio_clip = round(self.clip_w / self.clip_h, 2)
        self.pad_width = 0
        self.pad_height = 0
        if self.ratio_clip < self.ratio_target:
            # necessary to add vertical borders
            new_width = round(self.clip_h * self.ratio_target, 0)
            self.pad_width = round((new_width - self.clip_w) / 2, 0)
            clip = clip.std.AddBorders(left=self.pad_width, right=self.pad_width, top=0, bottom=0)
        elif self.ratio_clip > self.ratio_target:
            # necessary to add horizontal borders
            new_height = round(self.clip_w / self.ratio_target, 0)
            self.pad_height = round((new_height - self.clip_h) / 2, 0)
            clip = clip.std.AddBorders(left=0, right=0, top=self.pad_height, bottom=self.pad_height)
        # resize
        return clip.resize.Spline64(width=self.target_width, height=self.target_height)

    def restore_clip_size(self, clip: vs.VideoNode = None):
        clip = clip.resize.Spline64(width=self.clip_w + 2 * self.pad_width, height=self.clip_h + 2 * self.pad_height)
        if self.ratio_clip < self.ratio_target:
            # necessary to remove vertical borders
            clip = clip.std.Crop(left=self.pad_width, right=self.pad_width, top=0, bottom=0)
        elif self.ratio_clip > self.ratio_target:
            # necessary to remove horizontal borders
            clip = clip.std.Crop(left=0, right=0, top=self.pad_height, bottom=self.pad_height)
        return clip


    def clip_chroma_resize(self, clip_highres: vs.VideoNode, clip_lowres: vs.VideoNode) -> vs.VideoNode:
        clip_resized = self.restore_clip_size(clip_lowres)
        clip_bw = clip_highres.resize.Bicubic(format=vs.YUV420P8, matrix_s="709", range_s="full")
        clip_color = clip_resized.resize.Bicubic(format=vs.YUV420P8, matrix_s="709", range_s="full")
        clip_yuv = vs.core.std.ShufflePlanes(clips=[clip_bw, clip_color, clip_color], planes=[0, 1, 2], colorfamily=vs.YUV)
        return clip_yuv.resize.Bicubic(format=vs.RGB24, matrix_in_s="709", range_s="full", dither_type="error_diffusion")

class SmartResizeReference:
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, clip_size: list = [432, 768]):

        if not self._initialized:
            self.target_width = clip_size[1]
            self.target_height = clip_size[0]
            self.ratio_target = round(self.target_width / self.target_height, 2)
            self._initialized = True

    def get_resized_clip(self, clip: vs.VideoNode) -> vs.VideoNode:
        self.clip_w = clip.width
        self.clip_h = clip.height
        self.ratio_clip = round(self.clip_w / self.clip_h, 2)
        self.pad_width = 0
        self.pad_height = 0
        if self.ratio_clip < self.ratio_target:
            # necessary to add vertical borders
            new_width = round(self.clip_h * self.ratio_target, 0)
            self.pad_width = round((new_width - self.clip_w) / 2, 0)
            # pad_width must a multiple of 2
            self.pad_width = math.trunc(self.pad_width / 2) * 2
            clip = clip.std.AddBorders(left=self.pad_width, right=self.pad_width, top=0, bottom=0)
        elif self.ratio_clip > self.ratio_target:
            # necessary to add horizontal borders
            new_height = round(self.clip_w / self.ratio_target, 0)
            self.pad_height = round((new_height - self.clip_h) / 2, 0)
            # pad_height must a multiple of 2
            self.pad_height = math.trunc(self.pad_height / 2) * 2
            clip = clip.std.AddBorders(left=0, right=0, top=self.pad_height, bottom=self.pad_height)
        # resize
        return clip.resize.Spline64(width=self.target_width, height=self.target_height)

    def restore_clip_size(self, clip: vs.VideoNode = None):
        clip = clip.resize.Spline64(width=self.clip_w + 2 * self.pad_width, height=self.clip_h + 2 * self.pad_height)
        if self.ratio_clip < self.ratio_target:
            # necessary to remove vertical borders
            clip = clip.std.Crop(left=self.pad_width, right=self.pad_width, top=0, bottom=0)
        elif self.ratio_clip > self.ratio_target:
            # necessary to remove horizontal borders
            clip = clip.std.Crop(left=0, right=0, top=self.pad_height, bottom=self.pad_height)
        return clip


    def clip_chroma_resize(self, clip_highres: vs.VideoNode, clip_lowres: vs.VideoNode) -> vs.VideoNode:
        clip_resized = self.restore_clip_size(clip_lowres)
        clip_bw = clip_highres.resize.Bicubic(format=vs.YUV420P8, matrix_s="709", range_s="full")
        clip_color = clip_resized.resize.Bicubic(format=vs.YUV420P8, matrix_s="709", range_s="full")
        clip_yuv = vs.core.std.ShufflePlanes(clips=[clip_bw, clip_color, clip_color], planes=[0, 1, 2], colorfamily=vs.YUV)
        return clip_yuv.resize.Bicubic(format=vs.RGB24, matrix_in_s="709", range_s="full", dither_type="error_diffusion")
