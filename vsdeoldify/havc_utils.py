"""
------------------------------------------------------------------------------- 
Author: Dan64
Date: 2025-02-06
version: 
LastEditors: Dan64
LastEditTime: 2025-09-17
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
HAVC utility filter functions
"""
from __future__ import annotations
from functools import partial

import os
import pathlib

import warnings
import logging
import math
import cv2
import numpy as np

import torch
import functools
import math
import numpy as np
import cv2
from PIL import Image
from functools import partial

from vsdeoldify.vsslib.vsutils import frame_to_image, image_to_frame
from vsdeoldify.vsslib.vsfilters import *
import vsdeoldify.vsslib.restcolor as restcolor
from vsdeoldify.vsslib.imfilters import get_image_luma, image_luma_blend

from vapoursynth import core
import vapoursynth as vs

VIDEO_EXTENSIONS = ['.mpg', '.mp4', '.m4v', '.avi', '.mkv', '.mpeg']

"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description: 
------------------------------------------------------------------------------- 
function to read a video clip
"""


def HAVC_read_video(source: str, fpsnum: int = 0, fpsden: int = 1) -> vs.VideoNode:
    """HAVC utility function to read a video provided externally.
       The clip provided in output will be already in RGB24 format

    :param source:       Full path to the video to read
    :param fpsnum:       FPS numerator, for using it in HAVC, must be provided the
                         same value of clip to be colored: clip.fps_num
    :param fpsden:       FPS denominator, for using it in HAVC, must be provided the
                         same value of clip to be colored: clip.fps_den
    """
    if not os.path.isfile(source):
        HAVC_LogMessage(MessageType.EXCEPTION, "HAVC: invalid clip -> " + source)

    ext = source.lower()
    if not any(ext.endswith(extension) for extension in VIDEO_EXTENSIONS):
        HAVC_LogMessage(MessageType.EXCEPTION, "HAVC: invalid clip extension -> " + source)

    try:
        clip = vs.core.lsmas.LWLibavSource(source=source, stream_index=0, fpsnum=fpsnum, fpsden=fpsden,
                                           cache=0, prefer_hw=0)
    except Exception as error:
        HAVC_LogMessage(MessageType.EXCEPTION, "HAVC: LSMASHSource.dll not loaded or invalid clip -> " + str(error))

    # setting color matrix to 709.
    clip = vs.core.std.SetFrameProps(clip, _Matrix=vs.MATRIX_BT709)
    # setting color transfer (vs.TRANSFER_BT709), if it is not set.
    if _transferIsInvalid(clip):
        clip = core.std.SetFrameProps(clip=clip, _Transfer=vs.TRANSFER_BT709)
    # setting color primaries info (to vs.PRIMARIES_BT709), if it is not set.
    if _primariesIsInvalid(clip):
        clip = core.std.SetFrameProps(clip=clip, _Primaries=vs.PRIMARIES_BT709)
    # setting color range to TV (limited) range.
    clip = core.std.SetFrameProps(clip=clip, _ColorRange=vs.RANGE_LIMITED)
    # making sure frame rate is set
    clip = core.std.AssumeFPS(clip=clip, fpsnum=clip.fps_num, fpsden=clip.fps_den)
    # making sure the detected scan type is set (detected: progressive)
    clip = core.std.SetFrameProps(clip=clip, _FieldBased=vs.FIELD_PROGRESSIVE)  # progressive
    # changing range from limited to full range for HAVC
    clip = core.resize.Bicubic(clip, range_in_s="limited", range_s="full")
    # setting color range to PC (full) range.
    clip = core.std.SetFrameProps(clip=clip, _ColorRange=vs.RANGE_FULL)
    # adjusting color space to RGB24 for HAVC
    clip = core.resize.Bicubic(clip=clip, format=vs.RGB24, matrix_in_s="709", range_s="full")

    return clip


"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description: 
------------------------------------------------------------------------------- 
Utility functions for HAVC_main
"""


def _get_render_factors(Preset: str):
    # Select presets / tuning
    Preset = Preset.lower()
    presets = ['placebo', 'veryslow', 'slower', 'slow', 'medium', 'fast', 'faster', 'veryfast']
    preset0_rf = [32, 30, 28, 26, 24, 22, 20, 16]
    preset1_rf = [44, 36, 32, 28, 24, 22, 20, 16]

    pr_id = 5  # default 'fast'
    try:
        pr_id = presets.index(Preset)
    except ValueError:
        HAVC_LogMessage(MessageType.EXCEPTION, "HAVC_main: Preset choice is invalid for '" + str(pr_id) + "'")

    return pr_id, preset0_rf[pr_id], preset1_rf[pr_id]


def _get_mweight(VideoTune: str):
    # Select VideoTune
    VideoTune = VideoTune.lower()
    video_tune = ['verystable', 'morestable', 'stable', 'balanced', 'vivid', 'morevivid', 'veryvivid']
    ddcolor_weight = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    w_id = 3
    try:
        w_id = video_tune.index(VideoTune)
    except ValueError:
        HAVC_LogMessage(MessageType.EXCEPTION, "HAVC_main: VideoTune choice is invalid for '" + VideoTune + "'")

    return ddcolor_weight[w_id]


def _get_comb_method(CombMethod: str):
    # Select VideoTune
    CombMethod = CombMethod.lower()
    comb_str = ['simple', 'constrained-chroma', 'luma-masked', 'adaptive-luma']
    method_id = [2, 3, 4, 5]

    w_id = 2
    try:
        w_id = comb_str.index(CombMethod)
    except ValueError:
        HAVC_LogMessage(MessageType.EXCEPTION, "HAVC_main: CombMethod choice is invalid for '" + CombMethod + "'")

    return method_id[w_id]


def _get_color_model(ColorModel: str):
    ColorModel = ColorModel.lower()

    do_model = 0  # default: Video
    dd_model = 1  # default: Artistic

    if 'siggraph17' in ColorModel:
        dd_model = 2
    elif 'eccv16' in ColorModel:
        dd_model = 3

    if 'modelscope' in ColorModel:
        dd_model = 0
        if 'artistic' in ColorModel:
            do_model = 2
    if 'stable' in ColorModel:
        do_model = 1

    if '+' in ColorModel:
        dd_method = 2
    elif 'deoldify' in ColorModel:
        dd_method = 0
        if 'artistic' in ColorModel:
            do_model = 2
    elif 'ddcolor' in ColorModel:
        dd_method = 1
    else :
        dd_method = 1

    return do_model, dd_model, dd_method


def _get_color_tune(ColorTune: str, ColorFix: str, ColorMap: str, dd_model: int):
    # Select ColorTune for ColorFix
    if ColorTune is None:
        ColorTune = "none"
    ColorTune = ColorTune.lower()
    color_tune = ['none', 'light', 'medium', 'strong']
    if dd_model == 0:
        hue_tune = ["1.0,0.0", "0.7,0.1", "0.5,0.1", "0.2,0.1"]
    elif dd_model == 2:
        hue_tune = ["1.0,0.0", "0.6,0.1", "0.4,0.2", "0.2,0.1"]
    elif dd_model == 3:
        hue_tune = ["1.0,0.0", "0.7,0.1", "0.6,0.1", "0.3,0.1"]
    else:
        hue_tune = ["1.0,0.0", "0.8,0.1", "0.5,0.1", "0.2,0.1"]
    hue_tune2 = ["1.0,0.0", "0.9,0", "0.7,0", "0.5,0"]

    tn_id = 0
    try:
        tn_id = color_tune.index(ColorTune)
    except ValueError:
        HAVC_LogMessage(MessageType.EXCEPTION, "HAVC_main: ColorTune choice is invalid for '" + ColorTune + "'")

    # Select ColorFix for ddcolor/stabilizer
    if ColorFix is None:
        ColorFix = "none"

    ColorFix = ColorFix.lower()
    color_fix = ['none', 'magenta', 'magenta/violet', 'violet', 'violet/red', 'blue/magenta', 'yellow', 'yellow/orange',
                 'yellow/green']
    hue_fix = ["none", "270:300", "250:360", "300:330", "300:360", "220:280", "60:90", "30:90", "60:120"]

    co_id = 5
    try:
        co_id = color_fix.index(ColorFix)
    except ValueError:
        HAVC_LogMessage(MessageType.EXCEPTION, "HAVC_main: ColorFix choice is invalid for '" + ColorFix + "'")

    if co_id == 0 or tn_id == 0:
        hue_range = "none"
        hue_range2 = "none"
        dd_tweak = False  # in this case the Tweaks for DDcolor are disabled
    else:
        hue_range = hue_fix[co_id] + "|" + hue_tune[tn_id]
        hue_range2 = hue_fix[co_id] + "|" + hue_tune2[tn_id]
        dd_tweak = True  # in this case the Tweaks for DDcolor are enabled

    # Select Color Mapping
    ColorMap = ColorMap.lower()
    hue_w = ["1.0", "0.90", "0.80", "0.75"]
    colormap = ['none', 'blue->brown', 'blue->red', 'blue->green', 'green->brown', 'green->red', 'green->blue',
                'redrose->brown', 'redrose->blue', "red->brown", 'red->blue', 'yellow->rose']
    hue_map = ["none", "180:280|+140", "180:280|+100", "180:280|+220", "80:180|+260", "80:180|+220", "80:180|+140",
               "300:360,0:20|+40", "300:360,0:20|+260", "320:360|+50", "300:360|+260", "30:90|+300"]

    cl_id = 0
    try:
        cl_id = colormap.index(ColorMap)
    except ValueError:
        ret_range = restcolor._parse_hue_adjust(ColorMap)
        if ret_range is None:
            HAVC_LogMessage(MessageType.EXCEPTION, "HAVC_main: ColorMap choice is invalid for '" + ColorMap + "'")
        else:
            cl_id = -1

    if cl_id == 0:
        chroma_adjust = "none"
        chroma_adjust2 = "none"
    elif cl_id == -1:
        chroma_adjust = ColorMap
        chroma_adjust2 = "none"
    else:
        chroma_adjust = hue_map[cl_id] + "," + hue_w[tn_id]
        if tn_id == 0:
            chroma_adjust2 = "none"
        else:
            chroma_adjust2 = chroma_adjust

    return dd_tweak, hue_range, hue_range2, chroma_adjust, chroma_adjust2


def _check_input(DeepExOnlyRefFrames: bool, ScFrameDir: str, DeepExMethod: int, ScThreshold: float,
                 ScMinFreq: int, DeepExRefMerge: int):
    if DeepExOnlyRefFrames and (ScFrameDir is None):
        HAVC_LogMessage(MessageType.EXCEPTION, "HAVC_main: DeepExOnlyRefFrames is enabled but ScFrameDir is unset")

    if not (ScFrameDir is None) and DeepExMethod != 0 and DeepExOnlyRefFrames:
        HAVC_LogMessage(MessageType.EXCEPTION,
                        "HAVC_main: DeepExOnlyRefFrames is enabled but method not = 0 (HAVC)")

    if DeepExMethod != 0 and (ScFrameDir is None):
        HAVC_LogMessage(MessageType.EXCEPTION, "HAVC_main: DeepExMethod != 0 but ScFrameDir is unset")

    if DeepExMethod in (0, 1, 2, 5, 6) and ScThreshold == 0 and ScMinFreq == 0:
        HAVC_LogMessage(MessageType.EXCEPTION,
                        "HAVC_main: DeepExMethod in (0, 1, 2, 5, 6) but ScThreshold and ScMinFreq are not set")

    if DeepExMethod in (2, 6) and DeepExRefMerge > 0:
        HAVC_LogMessage(MessageType.EXCEPTION,
                        "HAVC_main: RefMerge cannot be used with DeepExMethod in (2, 6)")


# ------------------------------------------------------------
# collection of small helper functions to validate parameters
# ------------------------------------------------------------

def _matrixIsInvalid(clip: vs.VideoNode) -> bool:
    frame = clip.get_frame(0)
    value = frame.props.get('_Matrix', None)
    return value in [None, 2, 3] or value not in vs.MatrixCoefficients.__members__.values()


def _transferIsInvalid(clip: vs.VideoNode) -> bool:
    frame = clip.get_frame(0)
    value = frame.props.get('_Transfer', None)
    return value in [None, 0, 2, 3] or value not in vs.TransferCharacteristics.__members__.values()


def _primariesIsInvalid(clip: vs.VideoNode) -> bool:
    frame = clip.get_frame(0)
    value = frame.props.get('_Primaries', None)
    return value in [None, 2] or value not in vs.ColorPrimaries.__members__.values()


def _rangeIsInvalid(clip: vs.VideoNode) -> bool:
    frame = clip.get_frame(0)
    value = frame.props.get('_ColorRange', None)
    return value is None or value not in vs.ColorRange.__members__.values()


def _fieldBaseIsInvalid(clip: vs.VideoNode) -> bool:
    frame = clip.get_frame(0)
    value = frame.props.get('_FieldBased', None)
    return value is None or value not in vs.FieldBased.__members__.values()


def adjust_rgb(clip: vs.VideoNode, factor: list = (1.0, 1.0, 1.0), bias: list = (0, 0, 0),
               gamma: list = (1.0, 1.0, 1.0)) -> vs.VideoNode:
    """Utility function to change the color and luminance of RGB clip.
       Gain, bias (offset) and gamma can be set independently on each channel.

       :param clip:         Clip to process. Only RGB24 format is supported.
       :param factor:       List of Red, green and blue scaling factor, in the list format: (r, g, b).
                            Range 0.0 to 255.0, default = (1, 1, 1).
                            For example, r=1.3 multiplies the red channel pixel values by 1.3.
       :param bias:         List of Red, green and blue bias adjustments, in the list format: (rb, gb, bb).
                            Bias adjustment—add a fixed positive or negative value to a channel's pixel values.
                            For example, rb=16 will add 16 to all red pixel values and rb=-32 will subtract 32 from all
                            red pixel values, default = (0, 0, 0).
       :param gamma:        List of Red, green and blue gamma adjustments, in the list format: (rg, gg, bg).
                            Gamma adjustment—an exponential gain factor. For example, rg=1.2 will brighten the red
                            pixel values and gg=0.8 will darken the green pixel values.
    """
    funcName = 'HAVC_adjust_rgb'

    rgb = clip

    # unpack rgb_factor
    r = factor[0]
    g = factor[1]
    b = factor[2]

    # unpack rgb_bias
    rb = bias[0]
    gb = bias[1]
    bb = bias[2]

    # unpack rgb_gamma
    rg = gamma[0]
    gg = gamma[1]
    bg = gamma[2]

    if rgb.format.color_family != vs.RGB:
        raise ValueError(funcName + ': input clip needs to be RGB!')

    type = rgb.format.sample_type
    size = 2 ** rgb.format.bits_per_sample
    # adjusting bias values rb,gb,bb for any RGB bit depth
    limited = rgb.get_frame(0).props['_ColorRange'] == 1
    if limited:
        if rb > 235 or rb < -235:
            raise ValueError(funcName + ': source is flagged as "limited" but rb is out of range [-235,235]!')
        if gb > 235 or gb < -235:
            raise ValueError(funcName + ': source is flagged as "limited" but gb is out of range [-235,235]!')
        if bb > 235 or bb < -235:
            raise ValueError(funcName + ': source is flagged as "limited" but bb is out of range [-235,235]!')
    else:
        if rb > 255 or rb < -255:
            raise ValueError(funcName + ': source is flagged as "full" but rb is out of range [-255,255]!')
        if gb > 255 or gb < -255:
            raise ValueError(funcName + ': source is flagged as "limited" but gb is out of range [-235,235]!')
        if bb > 255 or bb < -255:
            raise ValueError(funcName + ': source is flagged as "limited" but bb is out of range [-235,235]!')

    if rg < 0:
        raise ValueError(funcName + ': rg needs to be >= 0!')
    if gg < 0:
        raise ValueError(funcName + ': gg needs to be >= 0!')
    if bg < 0:
        raise ValueError(funcName + ': bg needs to be >= 0!')

    if limited:
        if type == vs.INTEGER:
            maxVal = 235
        else:
            maxVal = 235.0
    else:
        if type == vs.INTEGER:
            maxVal = 255
        else:
            maxVal = 255.0
    rb, gb, bb = map(lambda b: b if size == maxVal else size / maxVal * b if type == vs.INTEGER else b / maxVal,
                     [rb, gb, bb])

    # x*r + rb , x*g + gb , x*b + bb
    rgb_adjusted = core.std.Expr(rgb, [f"x {r} * {rb} +", f"x {g} * {gb} +", f"x {b} * {bb} +"])

    # gamma per channel
    planes = [core.std.ShufflePlanes(rgb_adjusted, planes=p, colorfamily=vs.GRAY) for p in [0, 1, 2]]
    planes = [core.std.Levels(planes[p], gamma=g) if not g == 1 else planes[p] for p, g in enumerate([rg, gg, bg])]
    rgb_adjusted = core.std.ShufflePlanes(planes, planes=[0, 0, 0], colorfamily=vs.RGB)
    return rgb_adjusted

"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description: histogram equalization (i.e. auto levels) implementation in python
             using OpenCV
URL: https://pyimagesearch.com/2021/02/01/opencv-histogram-equalization-and-adaptive-histogram-equalization-clahe/
URL: https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html
------------------------------------------------------------------------------- 
"""

def rgb_equalizer(clip: vs.VideoNode, method: int = 0, clip_limit : float = 1.0, gridsize: int = 8,
                  strength: float = 0.5, weight3: float = 0.3, luma_blend: bool = True,
                  range_tv: bool = True) -> vs.VideoNode:
   """Histogram equalization implementation using OpenCV

   :param clip:         Clip to process (support only RGB24).
   :param method:       Method used to perform the histogram equalization.
                        Allowed values are:
                            0 : Apply Contrast Limited Adaptive Histogram Equalization on Luma [41.5 fps] (default)
                            1 : Apply Simple Histogram Equalization on all RGB channels [54.5 fps]
                            2 : Apply CLAHE on all RGB channels [37.5 fps]
                            3 : method=0 and method=1 are merged [34.5] 
                            4 : Automatic brightness and contrast optimization with ScaleAbs [51.5 fps]
                            5 : Automatic brightness and contrast optimization with saturation arithmetic [22.5 fps]
   :param clip_limit:   Threshold for contrast limiting, range [0, 50] (default=1.0)
   :param gridsize:     Size of grid for histogram equalization. The input image will be divided into equally
                        sized rectangular tiles. gridsize defines the number of tiles in row and column.
                        Used by models: 0, 2, 3 (default=8)
   :param strength:     Strength of the filter. A strength=0 means that the clip is returned unchanged,
                        range [0, 1] (default=0.5)
   :param weight3:      Weight for method 3 (default=0.5)
   :param luma_blend:   If enabled the equalized image is blended with the original image, darker is the image and more
                        weight will be assigned to the original image. default = True
   :param range_tv:     If True, the clip is using limited TV range.
   """

   rgb_clip = clip
   rgb_orig = rgb_clip

   # A zero weight means that the clip filtered is returned unchanged and 1 means that original clip is returned
   weight: float = min(max(1.0 - strength, 0.0), 1.0)

   #  Contrast Limited Adaptive Histogram Equalization
   def frame_autolevels_CLAHE_yuv(n, f, limit : float = 2.0, gridsize: int = 8, blend: bool = True):

      img = frame_to_image(f)
      img_np = np.asarray(img)

      yuv = cv2.cvtColor(img_np, cv2.COLOR_RGB2YUV)

      y_image, u_image, v_image = cv2.split(yuv)

      if range_tv:
        maxrange = 235
        minrange = 16
        f_luma = max(round(np.mean(y_image) / maxrange, 6) - 0.07, 0)
      else:
        maxrange = 255
        minrange = 0
        f_luma = round(np.mean(y_image) / maxrange, 6)

      f_luma_bright = DEF_THT_DARK_BLACK <= f_luma <= DEF_THT_BRIGHT_WHITE

      if not f_luma_bright:
        #HAVC_LogMessage(MessageType.WARNING, "HAVC_bw_tune: frame ", n, " luma: ", f_luma)
        return f.copy()

      clahe = cv2.createCLAHE(clipLimit=limit, tileGridSize=(gridsize, gridsize))
      y_image_eq = clahe.apply(y_image)
      yuv[:, :, 0] = y_image_eq.clip(min=minrange, max=maxrange).astype(int)
      img_new = Image.fromarray(cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB))

      """
      if f_luma < 0.35:
          bright_scale = min(max(pow(f_luma / 0.35, 3.0), 0), 1)
          w = round(max(0.90 * bright_scale, 0.15), 5)
          HAVC_LogMessage(MessageType.WARNING, "HAVC_bw_tune: frame ", n, " luma: ", f_luma, " weight: ", w)
      else:
          HAVC_LogMessage(MessageType.WARNING, "HAVC_bw_tune: frame ", n, " luma: ", f_luma)
      """

      img_m = image_luma_blend(img, img_new, f_luma, 0.40, 0.90, 0.35, 2.0) if blend else img_new

      return image_to_frame(img_m, f.copy())

   # CLAHE (Contrast Limited Adaptive Histogram Equalization) is used to improve the contrast of images
   def frame_autolevels_CLAHE_rgb(n, f, limit : float = 2.0, gridsize: int = 8, algo = 0, blend: bool = True):

      img = frame_to_image(f)

      if range_tv:
        f_luma = max(get_image_luma(img, 235) - 0.07, 0)
      else:
        f_luma = get_image_luma(img, 255)

      f_luma_bright = DEF_THT_DARK_BLACK <= f_luma <= DEF_THT_BRIGHT_WHITE

      if not f_luma_bright:
          return f.copy()

      img_np = np.asarray(img)
      r_image, g_image, b_image = cv2.split(img_np)

      if algo == 0:
         clahe = cv2.createCLAHE(clipLimit=limit, tileGridSize=(gridsize, gridsize))

         r_image_eq = clahe.apply(r_image)
         g_image_eq = clahe.apply(g_image)
         b_image_eq = clahe.apply(b_image)
      else:
         r_image_eq = cv2.equalizeHist(r_image)
         g_image_eq = cv2.equalizeHist(g_image)
         b_image_eq = cv2.equalizeHist(b_image)

      img_new = Image.fromarray(cv2.merge((r_image_eq, g_image_eq, b_image_eq)))

      """
      if f_luma < 0.40:
          bright_scale = min(max(pow(f_luma / 0.40, 4.0), 0), 1)
          w = round(max(0.90 * bright_scale, 0.10), 5)
          HAVC_LogMessage(MessageType.WARNING, "HAVC_bw_tune: frame ", n, " luma: ", f_luma, " weight: ", w)
      else:
          HAVC_LogMessage(MessageType.WARNING, "HAVC_bw_tune: frame ", n, " luma: ", f_luma)
      """

      img_m = image_luma_blend(img, img_new, f_luma, 0.40, 0.90, 0.15, 4.0) if blend else img_new

      return image_to_frame(img_m, f.copy())

   # Automatic brightness and contrast optimization with optional histogram clipping
   # https://stackoverflow.com/questions/56905592/automatic-contrast-and-brightness-adjustment-of-a-color-photo-of-a-sheet-of-pape
   def autolevels_with_Scale(n, f, clip_hist_percent: float = 1.0, algo: int = 0, blend: bool = True):

      img = frame_to_image(f)

      if range_tv:
          maxrange = 235
          minrange = 16
          f_luma = max(get_image_luma(img, maxrange) - 0.07, 0)
      else:
          maxrange = 255
          minrange = 0
          f_luma = get_image_luma(img, maxrange)

      f_luma_bright = DEF_THT_DARK_BLACK <= f_luma <= DEF_THT_BRIGHT_WHITE

      if not f_luma_bright:
          return f.copy()

      img_np = np.asarray(img)
      gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

      # Calculate grayscale histogram
      hist = cv2.calcHist([gray], [0], None, [maxrange+1], [0, maxrange+1])
      hist_size = len(hist)

      # Calculate cumulative distribution from the histogram
      accumulator = [float(hist[0])]
      for index in range(1, hist_size):
         accumulator.append(accumulator[index - 1] + float(hist[index]))

      # Locate points to clip
      maximum = accumulator[-1]
      clip_hist_percent *= (maximum / 100.0)
      clip_hist_percent /= 2.0

      # Locate left cut
      minimum_gray = 0
      while accumulator[minimum_gray] < clip_hist_percent:
         minimum_gray += 1

      # Locate right cut
      maximum_gray = hist_size - 1
      while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
         maximum_gray -= 1

      # Calculate alpha and beta values
      alpha = maxrange / max(maximum_gray - minimum_gray, 1)
      beta = -minimum_gray * alpha

      if algo == 0:
         img_np_new = cv2.convertScaleAbs(img_np, alpha=alpha, beta=beta)
      else:
         # Add bias and gain to an image with saturation arithmetics. Unlike cv2.convertScaleAbs,
         # it does not take an absolute value, which would lead to nonsensical results
         # e.g., a pixel at 44 with alpha = 3 and beta = -210 becomes 78 with OpenCV, when in fact it should become 0.
         img_np_new = (img_np * alpha + beta).clip(min=minrange, max=maxrange).astype(np.uint8)

      img_new = Image.fromarray(img_np_new)

      """
      if f_luma < 0.35:
          bright_scale = min(max(pow(f_luma / 0.35, 3.0), 0), 1)
          w = round(max(0.90 * bright_scale, 0.15), 5)
          HAVC_LogMessage(MessageType.WARNING, "HAVC_bw_tune: frame ", n, " luma: ", f_luma, " weight: ", w)
      else:
          HAVC_LogMessage(MessageType.WARNING, "HAVC_bw_tune: frame ", n, " luma: ", f_luma)
      """

      img_m = image_luma_blend(img, img_new, f_luma, 0.40, 0.90, 0.25, 3.0) if blend else img_new

      return image_to_frame(img_m, f.copy())

   if method == 0:
      clip_a = clip.std.ModifyFrame(clips=rgb_clip, selector=partial(frame_autolevels_CLAHE_yuv, limit=clip_limit,
                                                                     gridsize=gridsize, blend=luma_blend))
   elif method == 1:
      clip_a = clip.std.ModifyFrame(clips=rgb_clip, selector=partial(frame_autolevels_CLAHE_rgb, algo=1, limit=clip_limit,
                                                                     gridsize=gridsize, blend=luma_blend))
   elif method == 2:
      clip_a = clip.std.ModifyFrame(clips=rgb_clip, selector=partial(frame_autolevels_CLAHE_rgb, algo=0, limit=clip_limit,
                                                                     gridsize=gridsize, blend=luma_blend))
   elif method == 3:
      clip_a = clip.std.ModifyFrame(clips=rgb_clip, selector=partial(frame_autolevels_CLAHE_yuv, limit=clip_limit,
                                                                       gridsize=gridsize, blend=luma_blend))
      clip_b = clip.std.ModifyFrame(clips=rgb_clip, selector=partial(frame_autolevels_CLAHE_rgb, algo=1, limit=clip_limit,
                                                                      gridsize=gridsize, blend=luma_blend))
      # weight=0 means that is returned clip_a, weight=1 means that is returned clip_b
      clip_a = core.std.Merge(clip_a, clip_b, weight3)
   elif method ==4:
      clip_a = clip.std.ModifyFrame(clips=rgb_clip, selector=partial(autolevels_with_Scale, algo=0, blend=luma_blend,
                                                                     clip_hist_percent=clip_limit))
   else:
      clip_a = clip.std.ModifyFrame(clips=rgb_clip, selector=partial(autolevels_with_Scale, algo=1, blend=luma_blend,
                                                                     clip_hist_percent=clip_limit))

   # A zero weight means that clip_a is returned unchanged and 1 means that rgb_orig is returned unchanged
   if 0 <= weight < 1:
      clip_rgb = core.std.Merge(clip_a, rgb_orig, weight)
   else:
      clip_rgb = rgb_orig  # is returned the original clip

   if clip.format.id != vs.RGB24:
      # convert the format for tweak to YUV 8bits
      clip_new = clip_rgb.resize.Bicubic(format=vs.YUV420P8, matrix_s="709", range_s="limited")
   else:
      clip_new = clip_rgb

   return clip_new

"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description: function that takes a video clip as an input and calculates the average color values 
for each of the three color planes (red, green, blue), with optional RGB scaling factors.
The white balance of the input clip is adjusted based on the color balance of the individual frames.
URL: http://www.vapoursynth.com/doc/functions/frameeval.html
------------------------------------------------------------------------------- 
"""
def rgb_balance(clip: vs.VideoNode, strength: float = 0.5, rgb_factor: list = (1.0, 1.0, 1.0) ) -> vs.VideoNode:
   """ Auto white balance filter using PlaneStats()

    :param clip:           Clip to process (support only RGB24).
    :param strength:       Strength of the filter. A strength=0 means that the clip is returned unchanged,
                           range [0, 1] (default=0.5)
    :param rgb_factor:     List of Red, Green and Blue scaling factor, in the list format: (r, g, b),
                           default = (1, 1, 1). For example, r=1.3 multiplies the red channel pixel values by 1.3.

   """
   rgb_clip = clip

   # A zero weight means that the clip filtered is returned unchanged and 1 means that original clip is returned
   weight: float = min(max(1.0 - strength, 0.0), 1.0)

   # auto white from http://www.vapoursynth.com/doc/functions/frameeval.html
   def frame_autowhite(n, f, clip, core, rgb_fact):
      small_number = 0.000000001
      # unpack rgb_factor
      r = rgb_fact[0]
      g = rgb_fact[1]
      b = rgb_fact[2]
      red = f[0].props['PlaneStatsAverage']
      green = f[1].props['PlaneStatsAverage']
      blue = f[2].props['PlaneStatsAverage']
      max_rgb = max(red, green, blue)
      red_corr = max_rgb / max(red, small_number)
      green_corr = max_rgb / max(green, small_number)
      blue_corr = max_rgb / max(blue, small_number)
      norm = max(blue, math.sqrt(red_corr * red_corr + green_corr * green_corr + blue_corr * blue_corr) / math.sqrt(3),
                 small_number)
      r_gain = round(r * red_corr / norm, 8)
      g_gain = round(g * green_corr / norm, 8)
      b_gain = round(b * blue_corr / norm, 8)
      return core.std.Expr(clip,
                           expr=['x ' + repr(r_gain) + ' *', 'x ' + repr(g_gain) + ' *', 'x ' + repr(b_gain) + ' *'])

   r_avg = core.std.PlaneStats(rgb_clip, plane=0)
   g_avg = core.std.PlaneStats(rgb_clip, plane=1)
   b_avg = core.std.PlaneStats(rgb_clip, plane=2)

   clip_a = core.std.FrameEval(rgb_clip, functools.partial(frame_autowhite, clip=rgb_clip, core=vs.core,
                                                           rgb_fact=rgb_factor), prop_src=[r_avg, g_avg, b_avg])

   clip_b = rgb_clip

   # A zero weight means that clip_a is returned unchanged and 1 means that clip_b is returned unchanged
   if 0 <= weight < 1:
      clip_rgb = core.std.Merge(clip_a, clip_b, weight)
   else:
      clip_rgb = rgb_clip  # is returned the original clip

   if clip.format.id != vs.RGB24:
      # convert the format for tweak to YUV 8bits
      clip_new = clip_rgb.resize.Bicubic(format=vs.YUV420P8, matrix_s="709", range_s="limited")
   else:
      clip_new = clip_rgb

   return clip_new
