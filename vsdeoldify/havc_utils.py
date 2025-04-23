"""
------------------------------------------------------------------------------- 
Author: Dan64
Date: 2025-02-06
version: 
LastEditors: Dan64
LastEditTime: 2025-02-09
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
from PIL import Image

import torch

from vsdeoldify.vsslib.vsfilters import *

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


def adjust_rgb(rgb: vs.VideoNode, r: float = 1.0, g: float = 1.0, b: float = 1.0, rb: float = 0.0,
               gb: float = 0.0, bb: float = 0.0,  rg: float = 1.0, gg: float = 1.0, bg: float = 1.0) -> vs.VideoNode:
    funcName = 'HAVC'

    type = rgb.format.sample_type
    size = 2 ** rgb.format.bits_per_sample
    # adjusting bias values rb,gb,bb for any RGB bit depth
    limited = rgb.get_frame(0).props['_ColorRange'] == 1
    if limited:
        if rb > 235 or rb < -235: raise ValueError(
            funcName + ': source is flagged as "limited" but rb is out of range [-235,235]!')
        if gb > 235 or gb < -235: raise ValueError(
            funcName + ': source is flagged as "limited" but gb is out of range [-235,235]!')
        if bb > 235 or bb < -235: raise ValueError(
            funcName + ': source is flagged as "limited" but bb is out of range [-235,235]!')
    else:
        if rb > 255 or rb < -255: raise ValueError(
            funcName + ': source is flagged as "full" but rb is out of range [-255,255]!')
        if gb > 255 or gb < -255: raise ValueError(
            funcName + ': source is flagged as "limited" but gb is out of range [-235,235]!')
        if bb > 255 or bb < -255: raise ValueError(
            funcName + ': source is flagged as "limited" but bb is out of range [-235,235]!')

    if rg < 0: raise ValueError(funcName + ': rg needs to be >= 0!')
    if gg < 0: raise ValueError(funcName + ': gg needs to be >= 0!')
    if bg < 0: raise ValueError(funcName + ': bg needs to be >= 0!')

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

