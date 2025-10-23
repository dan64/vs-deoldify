"""
-------------------------------------------------------------------------------
Author: Dan64
Date: 2024-05-15
version:
LastEditors: Dan64
LastEditTime: 2025-10-19
-------------------------------------------------------------------------------
Description:
-------------------------------------------------------------------------------
Vapoursynth Smart Resize class.
"""

import vapoursynth as vs
import math
from typing import Optional

from vsdeoldify.vsslib.constants import DEF_MAX_RESIZE

"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
function to resize a clip by keeping the aspect ratio. 
"""


def resize_to_width(clip: vs.VideoNode, target_width: int = 512) -> vs.VideoNode:
    """
    Resize clip to target width while maintaining aspect ratio and ensuring
    height is divisible by 2 (required for many codecs and filters).

    Args:
        clip: Input clip
        target_width: Target width (default: 512)

    Returns:
        Resized clip with target_width and proportional height (divisible by 2)
    """
    # Calculate the proportional height
    target_height = round(clip.height * target_width / clip.width)

    # Ensure height is divisible by 2
    if target_height % 2 != 0:
        target_height += 1  # or -= 1, but +1 is generally safer to avoid undersizing

    # Resize using spline resampling
    resized_clip = clip.resize.Spline36(width=target_width, height=target_height)

    return resized_clip

def resize_to_chroma(clip_highres: vs.VideoNode, clip_lowres: vs.VideoNode) -> vs.VideoNode:
    """
        Perform a chroma Resize. The lowres clip will be resized to highres and the Y plane of clip_lowres
        will be replaced by the Y plane of highres clip.

        Args:
            clip_highres: Input highres clip with original plane Y
            clip_lowres: Input lowres clip to apply the chroma resize

        Returns:
            highres clip in RGB24 format with chroma resize
    """
    # perform resize if needed
    if clip_highres.width != clip_lowres.width or clip_highres.height != clip_lowres.height:
        clip_resized = clip_lowres.resize.Spline36(width=clip_highres.width, height=clip_highres.height)
    else:
        clip_resized = clip_lowres
    # convert clips to YUV
    clip_bw = clip_highres.resize.Bicubic(format=vs.YUV420P8, matrix_s="709", range_s="full")
    clip_color = clip_resized.resize.Bicubic(format=vs.YUV420P8, matrix_s="709", range_s="full")
    # restore orginal Y plane
    clip_yuv = vs.core.std.ShufflePlanes(clips=[clip_bw, clip_color, clip_color], planes=[0, 1, 2], colorfamily=vs.YUV)
    # convert result to RGB24
    return clip_yuv.resize.Bicubic(format=vs.RGB24, matrix_in_s="709", range_s="full", dither_type="error_diffusion")



"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
Class for resize clips with padding to keep the aspect ratio, 
borders are added if needed. Example of usage

padder = ClipPadder()  # default to 512x512

# Load and pad clip
clip_rgb = core.ffms2.Source("input.avi")
if clip_rgb.format.id != vs.RGB24:
    clip_rgb = core.resize.Bicubic(clip_rgb, format=vs.RGB24)

clip_padded = padder.pad(clip_rgb)  # Now 512x512

# Run coloring models (DDColor, DeOldify, etc.)
clip_processed = your_colorization_pipeline(clip_padded)

# Restore original resolution
clip_final = padder.unpad(clip_processed)

clip_final.set_output()

"""

class ClipPadder:
    """
    A class to pad a VapourSynth RGB clip to 512x512 and later restore it.
    Stores all necessary metadata internally.
    """

    def __init__(self, clip_width_size: int = DEF_MAX_RESIZE):
        self.clip_width_size: int = clip_width_size
        self._original_width: Optional[int] = None
        self._original_height: Optional[int] = None
        self._pad_left: Optional[int] = None
        self._pad_right: Optional[int] = None
        self._pad_top: Optional[int] = None
        self._pad_bottom: Optional[int] = None
        self._scale: Optional[float] = None
        self._is_padded: bool = False

    def pad(self, clip: vs.VideoNode) -> vs.VideoNode:
        """
        Pad input RGB clip to 512x512 with gray borders.
        Stores internal parameters for later unpadding.
        """
        if clip.format.color_family != vs.RGB:
            raise ValueError("Input clip must be RGB")

        w, h = clip.width, clip.height
        self._original_width = w
        self._original_height = h

        # Fit to 512x512 box (preserve aspect ratio)
        scale = self.clip_width_size / max(w, h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        self._scale = scale

        # Resize
        resized = vs.core.resize.Lanczos(clip, width=new_w, height=new_h)

        # Compute symmetric padding
        pad_w = self.clip_width_size - new_w
        pad_h = self.clip_width_size - new_h
        self._pad_left = pad_w // 2
        self._pad_right = pad_w - self._pad_left
        self._pad_top = pad_h // 2
        self._pad_bottom = pad_h - self._pad_top

        # Pad with neutral gray (128, 128, 128)
        padded = vs.core.std.AddBorders(
            resized,
            left=self._pad_left,
            right=self._pad_right,
            top=self._pad_top,
            bottom=self._pad_bottom,
            color=[128, 128, 128]
        )

        self._is_padded = True
        return padded

    def unpad(self, clip: vs.VideoNode) -> vs.VideoNode:
        """
        Restore clip to original resolution.
        Must be called after `pad()`.
        """
        if not self._is_padded:
            raise RuntimeError("unpad() called before pad()")

        if clip.width != self.clip_width_size or clip.height != self.clip_width_size:
            raise ValueError("Input to unpad() must be 512x512")

        # Crop back to content area (after padding)
        cropped = vs.core.std.Crop(
            clip,
            left=self._pad_left,
            top=self._pad_top,
            right=self._pad_right,
            bottom=self._pad_bottom
        )

        # Resize back to original resolution
        restored = vs.core.resize.Lanczos(
            cropped,
            width=self._original_width,
            height=self._original_height
        )

        return restored

    # Optional: expose metadata (read-only)
    @property
    def original_size(self) -> tuple[int, int]:
        if self._original_width is None:
            raise RuntimeError("pad() not called yet")
        return (self._original_width, self._original_height)

    @property
    def padding(self) -> tuple[int, int, int, int]:
        """(left, top, right, bottom)"""
        if self._pad_left is None:
            raise RuntimeError("pad() not called yet")
        return (self._pad_left, self._pad_top, self._pad_right, self._pad_bottom)

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
    _initialized: bool = False
    ex_model: int = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, clip_size: list = [432, 768], ex_model: int = 1):

        self.pad_width = None
        self.pad_height = None
        self.ratio_clip = None
        self.clip_h = None
        self.clip_w = None
        if not self.__class__._initialized:
            self.target_width = clip_size[1]
            self.target_height = clip_size[0]
            self.ex_model = ex_model
            self.ratio_target = round(self.target_width / self.target_height, 2)
            self.__class__._initialized = True

    def get_resized_clip(self, clip: vs.VideoNode) -> vs.VideoNode:
        self.clip_w = clip.width
        self.clip_h = clip.height
        self.ratio_clip = round(self.clip_w / self.clip_h, 2)
        self.pad_width = 0
        self.pad_height = 0
        if self.ex_model in (0, 1):
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
        else:
            return clip  # no changes

        # resize
        return clip.resize.Spline64(width=self.target_width, height=self.target_height)

    def restore_clip_size(self, clip: vs.VideoNode = None):
        if self.ex_model in (0, 1):
            clip = clip.resize.Spline64(width=self.clip_w + 2 * self.pad_width, height=self.clip_h + 2 * self.pad_height)
            if self.ratio_clip < self.ratio_target:
                # necessary to remove vertical borders
                clip = clip.std.Crop(left=self.pad_width, right=self.pad_width, top=0, bottom=0)
            elif self.ratio_clip > self.ratio_target:
                # necessary to remove horizontal borders
                clip = clip.std.Crop(left=0, right=0, top=self.pad_height, bottom=self.pad_height)
            return clip
        else:
            return clip  # no need to restore

    def clip_chroma_resize(self, clip_highres: vs.VideoNode, clip_lowres: vs.VideoNode) -> vs.VideoNode:
        clip_resized = self.restore_clip_size(clip_lowres)
        clip_bw = clip_highres.resize.Bicubic(format=vs.YUV420P8, matrix_s="709", range_s="full")
        clip_color = clip_resized.resize.Bicubic(format=vs.YUV420P8, matrix_s="709", range_s="full")
        clip_yuv = vs.core.std.ShufflePlanes(clips=[clip_bw, clip_color, clip_color], planes=[0, 1, 2],
                                             colorfamily=vs.YUV)
        return clip_yuv.resize.Bicubic(format=vs.RGB24, matrix_in_s="709", range_s="full",
                                       dither_type="error_diffusion")


class SmartResizeReference:
    _instance = None
    _initialized: bool = False
    ex_model: int = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, clip_size: list = [432, 768], ex_model: int = 1):

        self.pad_width = None
        self.pad_height = None
        self.ratio_clip = None
        self.clip_h = None
        self.clip_w = None
        if not self.__class__._initialized:
            self.target_width = clip_size[1]
            self.target_height = clip_size[0]
            self.ex_model = ex_model
            self.ratio_target = round(self.target_width / self.target_height, 2)
            self.__class__._initialized = True

    def get_resized_clip(self, clip: vs.VideoNode) -> vs.VideoNode:
        self.clip_w = clip.width
        self.clip_h = clip.height
        self.ratio_clip = round(self.clip_w / self.clip_h, 2)
        self.pad_width = 0
        self.pad_height = 0
        if self.ex_model in (0, 1):
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
        else:
            return clip  # no changes

    def restore_clip_size(self, clip: vs.VideoNode = None):
        if self.ex_model in (0, 1):
            clip = clip.resize.Spline64(width=self.clip_w + 2 * self.pad_width, height=self.clip_h + 2 * self.pad_height)
            if self.ratio_clip < self.ratio_target:
                # necessary to remove vertical borders
                clip = clip.std.Crop(left=self.pad_width, right=self.pad_width, top=0, bottom=0)
            elif self.ratio_clip > self.ratio_target:
                # necessary to remove horizontal borders
                clip = clip.std.Crop(left=0, right=0, top=self.pad_height, bottom=self.pad_height)
            return clip
        else:
            return clip  # no need to restore

    def clip_chroma_resize(self, clip_highres: vs.VideoNode, clip_lowres: vs.VideoNode) -> vs.VideoNode:
        clip_resized = self.restore_clip_size(clip_lowres)
        clip_bw = clip_highres.resize.Bicubic(format=vs.YUV420P8, matrix_s="709", range_s="full")
        clip_color = clip_resized.resize.Bicubic(format=vs.YUV420P8, matrix_s="709", range_s="full")
        clip_yuv = vs.core.std.ShufflePlanes(clips=[clip_bw, clip_color, clip_color], planes=[0, 1, 2],
                                             colorfamily=vs.YUV)
        return clip_yuv.resize.Bicubic(format=vs.RGB24, matrix_in_s="709", range_s="full",
                                       dither_type="error_diffusion")
