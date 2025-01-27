"""
------------------------------------------------------------------------------- 
Author: Dan64
Date: 2024-02-29
version: 
LastEditors: Dan64
LastEditTime: 2025-01-19
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
main Vapoursynth wrapper to pytorch-based coloring filter HybridAVC (HAVC).
The filter includes some portions of code from the following coloring projects:
DeOldify: https://github.com/jantic/DeOldify
DDColor: https://github.com/HolyWu/vs-ddcolor
Colorization: https://github.com/richzhang/colorization
Deep-Exemplar: https://github.com/zhangmozhe/Deep-Exemplar-based-Video-Colorization
ColorMNet: https://github.com/yyang181/colormnet
"""
from __future__ import annotations
from functools import partial

import os
import pathlib

os.environ["CUDA_MODULE_LOADING"] = "LAZY"
os.environ["NUMEXPR_MAX_THREADS"] = "8"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TORCH_LOGS"] = "-all"

import math
import cv2
import numpy as np
from PIL import Image

from vsdeoldify.deoldify import device
from vsdeoldify.deoldify.device_id import DeviceId
from vsdeoldify.vsslib.constants import *
from vsdeoldify.vsslib.vsfilters import *
from vsdeoldify.vsslib.mcomb import *
from vsdeoldify.vsslib.vsmodels import *
from vsdeoldify.vsslib.vsresize import SmartResizeColorizer, SmartResizeReference
from vsdeoldify.vsslib.vsscdect import SceneDetectFromDir, SceneDetect, CopySCDetect, get_sc_props

from vsdeoldify.deepex import deepex_colorizer, get_deepex_size, ModelColorizer

__version__ = "4.6.8"

import warnings
import logging

warnings.filterwarnings("ignore", category=UserWarning, message=".*?Your .*? set is empty.*?")
warnings.filterwarnings("ignore", category=UserWarning,
                        message="The parameter 'pretrained' is deprecated since 0.13 and may be removed in the "
                                "future, please use 'weights' instead.")
warnings.filterwarnings("ignore", category=UserWarning, message="Arguments other than a weight enum or `None`.*?")
warnings.filterwarnings("ignore", category=UserWarning, message="torch.nn.utils.weight_norm is deprecated.*?")
warnings.filterwarnings("ignore", category=UserWarning, message="Conversion from CIE-LAB,*?")
warnings.filterwarnings("ignore", category=UserWarning, message=".*?Torch was not compiled with flash attention.*?")

warnings.filterwarnings("ignore", category=FutureWarning, message=".torch.cuda.amp.custom_fwd.*?")
warnings.filterwarnings("ignore", category=FutureWarning, message="Arguments other than a weight enum or `None`.*?")
warnings.filterwarnings("ignore", category=FutureWarning, message=".You are using `torch.load`.*?")

warnings.simplefilter(action='ignore', category=FutureWarning)

package_dir = os.path.dirname(os.path.realpath(__file__))
model_dir = os.path.join(package_dir, "models")

# configuring torch
torch.backends.cudnn.benchmark = True

import vapoursynth as vs

"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
wrapper to HAVC filter with "presets" management
"""


def HAVC_main(clip: vs.VideoNode, Preset: str = 'Fast', ColorModel: str = 'Video+Artistic', VideoTune: str = 'Stable',
              ColorFix: str = 'Violet/Red', ColorTune: str = 'Light', ColorMap: str = 'None',
              EnableDeepEx: bool = False,
              DeepExMethod: int = 0, DeepExPreset: str = 'Medium', DeepExRefMerge: int = 0,
              DeepExOnlyRefFrames: bool = False,
              ScFrameDir: str = None, ScThreshold: float = DEF_THRESHOLD, ScThtOffset: int = 1, ScMinFreq: int = 0,
              ScMinInt: int = 1, ScThtSSIM: float = 0.0, ScNormalize: bool = True, DeepExModel: int = 0,
              DeepExVivid: bool = True, DeepExEncMode: int = 0, DeepExMaxMemFrames=0,
              enable_fp16: bool = True, sc_debug: bool = False) -> vs.VideoNode:
    """Main HAVC function supporting the Presets

    :param clip:                clip to process, only RGB24 format is supported.
    :param Preset:              Preset to control the encoding speed/quality.
                                Allowed values are:
                                    'Placebo',
                                    'VerySlow',
                                    'Slower',
                                    'Slow',
                                    'Medium',
                                    'Fast',  (default)
                                    'Faster',
                                    'VeryFast'
    :param ColorModel:          Preset to control the Color Models to be used for the color inference
                                Allowed values are:
                                    'Video+Artistic'  (default)
                                    'Video+ModelScope'
                                    'Video+Siggraph17'
                                    'Video+ECCV16'
                                    'DeOldify(Video)'
                                    'DDColor(Artistic)'
                                    'DDColor(ModelScope)'
                                    'Zhang(Siggraph17)'
                                    'Zhang(ECCV16)'
    :param VideoTune:           Preset to control the output video color stability
                                Allowed values are:
                                    'VeryStable',
                                    'MoreStable'
                                    'Stable',
                                    'Balanced',
                                    'Vivid',
                                    ,MoreVivid'
                                    'VeryVivid',
    :param ColorFix:            This parameter allows to reduce color noise on specific chroma ranges.
                                Allowed values are:
                                    'None',
                                    'Magenta',
                                    'Magenta/Violet',
                                    'Violet',
                                    'Violet/Red', (default)
                                    'Blue/Magenta',
                                    'Yellow',
                                    'Yellow/Orange',
                                    'Yellow/Green'
    :param ColorTune:           This parameter allows to define the intensity of noise reduction applied by ColorFix.
                                Allowed values are:
                                    'Light',  (default)
                                    'Medium',
                                    'Strong',
    :param ColorMap:            This parameter allows to change a given color range to another color.
                                Allowed values are:
                                    'None', (default)
                                    'Blue->Brown',
                                    'Blue->Red',
                                    'Blue->Green',
                                    'Green->Brown',
                                    'Green->Red',
                                    'Green->Blue',
                                    'Red->Brown',
                                    'Red->Blue'
                                    'Yellow->Rose'
    :param EnableDeepEx:        Enable coloring using "Exemplar-based" Video Colorization models
    :param DeepExMethod:        Method to use to generate reference frames.
                                        0 = HAVC same as video (default)
                                        1 = HAVC + RF same as video
                                        2 = HAVC + RF different from video
                                        3 = external RF same as video
                                        4 = external RF different from video
                                        5 = HAVC different from video
    :param DeepExPreset:        Preset to control the render method and speed:
                                Allowed values are:
                                        'Fast'   (colors are more washed out)
                                        'Medium' (colors are a little washed out)
                                        'Slow'   (colors are a little more vivid)
    :param DeepExRefMerge:      Method used by DeepEx to merge the reference frames with the frames propagated by DeepEx.
                                It is applicable only with DeepEx method: 0, 1, 2.
                                Allowed values are:
                                        0 = No RF merge (reference frames can be produced with any frequency)
                                        1 = RF-Merge VeryLow (reference frames are merged with weight=0.3)
                                        2 = RF-Merge Low (reference frames are merged with weight=0.4)
                                        3 = RF-Merge Med (reference frames are merged with weight=0.5)
                                        4 = RF-Merge High (reference frames are merged with weight=0.6)
                                        5 = RF-Merge VeryHigh (reference frames are merged with weight=0.7)
    :param DeepExOnlyRefFrames: If enabled the filter will output in "ScFrameDir" the reference frames. Useful to check
                                and eventually correct the frames with wrong colors
                                (can be used only if DeepExMethod in [0,5])
    :param DeepExModel:         Exemplar Model used by DeepEx to propagate color frames.
                                        0 : ColorMNet (default)
                                        1 : Deep-Exemplar
    :param DeepExVivid:         Depending on selected DeepExModel, if enabled (True):
                                    0) ColorMNet: the frames memory is reset at every reference frame update
                                    1) Deep-Exemplar: the saturation will be increased by about 25%.
                                range [True, False]
    :param DeepExEncMode:       Parameter used by ColorMNet to define the encode mode strategy.
                                Available values are:
                                     0: remote encoding. The frames will be colored by a thread outside Vapoursynth.
                                                         This option don't have any GPU memory limitation and will allow
                                                         to fully use the long term frame memory.
                                                         It is the faster encode method (default)
                                     1: local encoding.  The frames will be colored inside the Vapoursynth environment.
                                                         In this case the max_memory will be limited by the size of GPU
                                                         memory (max 15 frames for 24GB GPU).
                                                         Useful for coloring clips with a lot of smooth transitions,
                                                         since in this case is better to use a short frame memory or
                                                         the Deep-Exemplar model, which is faster.
    :param DeepExMaxMemFrames:  Parameter used by ColorMNet model, specify the max number of encoded frames to keep in memory.
                                Its value depend on encode mode and must be defined manually following the suggested values.
                                DeepExEncMode=0: there is no memory limit (it could be all the frames in the clip).
                                Suggested values are:
                                    min=150, max=10000
                                If = 0 will be filled with the value of 10000 or the clip length if lower.
                                DeepExEncMode=1: the max memory frames is limited by available GPU memory.
                                Suggested values are:
                                    min=1, max=4      : for 8GB GPU
                                    min=1, max=8      : for 12GB GPU
                                    min=1, max=15     : for 24GB GPU
                                If = 0 will be filled with the max value (depending on total GPU RAM available)
    :param ScFrameDir:          if set, define the directory where are stored the reference frames that will be used
                                by "Exemplar-based" Video Colorization models.
    :param ScThreshold:         Scene change threshold used to generate the reference frames to be used by
                                "Exemplar-based" Video Colorization. It is a percentage of the luma change between
                                the previous and the current frame. range [0-1], default 0.10. If =0 are not generate
                                reference frames.
    :param ScThtOffset:         Offset index used for the Scene change detection. The comparison will be performed,
                                between frame[n] and frame[n-offset]. An offset > 1 is useful to detect blended scene
                                change, range[1, 25]. Default = 1.
    :param ScMinInt:            Minimum number of frame interval between scene changes, range[1, 25]. Default = 1.
    :param ScMinFreq:           if > 0 will be generated at least a reference frame every "ScMinFreq" frames.
                                range [0-1500], default: 0.
    :param ScThtSSIM:           Threshold used by the SSIM (Structural Similarity Index Metric) selection filter.
                                If > 0, will be activated a filter that will improve the scene-change detection,
                                by discarding images that are similar.
                                Suggested values are between 0.35 and 0.65, range [0-1], default 0.0 (deactivated)
    :param ScNormalize:         If true the B&W frames are normalized before use misc.SCDetect(), the normalization will
                                increase the sensitivity to smooth scene changes, range [True, False], default: True
    :param enable_fp16:         Enable/disable FP16 in ddcolor inference, range [True, False]
    :param sc_debug:            Print debug messages regarding the scene change detection process.
    """
    # disable packages warnings
    disable_warnings()

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

    deoldify_rf = preset0_rf[pr_id]
    ddcolor_rf = preset1_rf[pr_id]

    # vs.core.log_message(2, "Preset index: " + str(pr_id) )

    # Select VideoTune
    VideoTune = VideoTune.lower()
    video_tune = ['verystable', 'morestable', 'stable', 'balanced', 'vivid', 'morevivid', 'veryvivid']
    ddcolor_weight = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    w_id = 3
    try:
        w_id = video_tune.index(VideoTune)
    except ValueError:
        HAVC_LogMessage(MessageType.EXCEPTION, "HAVC_main: VideoTune choice is invalid for '" + VideoTune + "'")

    ColorModel = ColorModel.lower()
    if 'siggraph17' in ColorModel:
        dd_model = 2
    elif 'eccv16' in ColorModel:
        dd_model = 3
    elif 'modelscope' in ColorModel:
        dd_model = 0
    else:
        dd_model = 1

    if 'deoldify' in ColorModel:
        dd_method = 0
    elif 'ddcolor' in ColorModel:
        dd_method = 1
    elif 'zhang' in ColorModel:
        dd_method = 1
    else:
        dd_method = 2

    # Select ColorTune for ColorFix
    ColorTune = ColorTune.lower()
    color_tune = ['light', 'medium', 'strong']
    if dd_model == 0:
        hue_tune = ["0.7,0.1", "0.5,0.1", "0.2,0.1"]
    elif dd_model == 2:
        hue_tune = ["0.6,0.1", "0.4,0.2", "0.2,0.1"]
    elif dd_model == 3:
        hue_tune = ["0.7,0.1", "0.6,0.1", "0.3,0.1"]
    else:
        hue_tune = ["0.8,0.1", "0.5,0.1", "0.2,0.1"]
    hue_tune2 = ["0.9,0", "0.7,0", "0.5,0"]

    tn_id = 0
    try:
        tn_id = color_tune.index(ColorTune)
    except ValueError:
        HAVC_LogMessage(MessageType.EXCEPTION, "HAVC_main: ColorTune choice is invalid for '" + ColorTune + "'")

    # Select ColorFix for ddcolor/stabilizer
    ColorFix = ColorFix.lower()
    color_fix = ['none', 'magenta', 'magenta/violet', 'violet', 'violet/red', 'blue/magenta', 'yellow', 'yellow/orange',
                 'yellow/green']
    hue_fix = ["none", "270:300", "270:330", "300:330", "300:360", "220:280", "60:90", "30:90", "60:120"]

    co_id = 5
    try:
        co_id = color_fix.index(ColorFix)
    except ValueError:
        HAVC_LogMessage(MessageType.EXCEPTION, "HAVC_main: ColorFix choice is invalid for '" + ColorFix + "'")

    if co_id == 0:
        hue_range = "none"
        hue_range2 = "none"
        dd_tweak = False  # in this case the Tweaks for DDcolor are disabled
    else:
        hue_range = hue_fix[co_id] + "|" + hue_tune[tn_id]
        hue_range2 = hue_fix[co_id] + "|" + hue_tune2[tn_id]
        dd_tweak = True  # in this case the Tweaks for DDcolor are enabled

    # Select Color Mapping
    ColorMap = ColorMap.lower()
    hue_w = ["0.90", "0.80", "0.70"]
    colormap = ['none', 'blue->brown', 'blue->red', 'blue->green', 'green->brown', 'green->red', 'green->blue',
                'redrose->brown', 'redrose->blue', "red->brown", 'yellow->rose']
    hue_map = ["none", "180:280|+140", "180:280|+100", "180:280|+220", "80:180|+260", "80:180|+220",
               "80:180|+140", "300:360,0:20|+40", "300:360,0:20|+260", "320:360,0:15|+50", "30:90|+300"]

    cl_id = 0
    try:
        cl_id = colormap.index(ColorMap)
    except ValueError:
        HAVC_LogMessage(MessageType.EXCEPTION, "HAVC_main: ColorMap choice is invalid for '" + ColorMap + "'")

    if cl_id == 0:
        chroma_adjust = "none"
        chroma_adjust2 = "none"
    else:
        chroma_adjust = hue_map[cl_id] + "," + hue_w[tn_id]
        if tn_id == 0:
            chroma_adjust2 = "none"
        else:
            chroma_adjust2 = chroma_adjust

    if EnableDeepEx and DeepExMethod in (0, 1, 2, 5):

        if DeepExOnlyRefFrames and (ScFrameDir is None):
            HAVC_LogMessage(MessageType.EXCEPTION, "HAVC_main: DeepExOnlyRefFrames is enabled but ScFrameDir is unset")

        if not (ScFrameDir is None) and DeepExMethod not in (0, 5) and DeepExOnlyRefFrames:
            HAVC_LogMessage(MessageType.EXCEPTION,
                            "HAVC_main: DeepExOnlyRefFrames is enabled but method not in (0, 5) (HAVC)")

        if DeepExMethod not in (0, 5) and (ScFrameDir is None):
            HAVC_LogMessage(MessageType.EXCEPTION, "HAVC_main: DeepExMethod not in (0, 5) but ScFrameDir is unset")

        if DeepExMethod in (0, 1, 2, 5) and ScThreshold == 0 and ScMinFreq == 0:
            HAVC_LogMessage(MessageType.EXCEPTION,
                            "HAVC_main: DeepExMethod in (0, 1, 2, 5) but ScThreshold and ScMinFreq are not set")

        if DeepExMethod in (2, 5) and DeepExRefMerge > 0:
            HAVC_LogMessage(MessageType.EXCEPTION,
                            "HAVC_main: RefMerge cannot be used with DeepExMethod in (2, 5)")

        ref_tresh = None
        if DeepExRefMerge > 0:
            ScMinFreq = 1
            if ScThreshold is not None and 0 < ScThreshold < 1:
                ref_tresh = ScThreshold
            else:
                ref_tresh = DEF_THRESHOLD

        clip_ref = HAVC_ddeoldify(clip, method=dd_method, mweight=ddcolor_weight[w_id],
                                  deoldify_p=[0, deoldify_rf, 1.0, 0.0],
                                  ddcolor_p=[dd_model, ddcolor_rf, 1.0, 0.0, enable_fp16],
                                  ddtweak=dd_tweak, ddtweak_p=[0.0, 1.0, 2.5, True, 0.3, 0.6, 1.5, 0.5, hue_range],
                                  sc_threshold=ScThreshold, sc_tht_offset=ScThtOffset, sc_min_freq=ScMinFreq,
                                  sc_min_int=ScMinInt, sc_tht_ssim=ScThtSSIM, sc_normalize=ScNormalize,
                                  sc_debug=sc_debug)

        clip_colored = HAVC_deepex(clip=clip, clip_ref=clip_ref, method=DeepExMethod, render_speed=DeepExPreset,
                                   render_vivid=DeepExVivid, ref_merge=DeepExRefMerge, sc_framedir=ScFrameDir,
                                   only_ref_frames=DeepExOnlyRefFrames, dark=True, dark_p=[0.2, 0.8],
                                   ref_thresh=ref_tresh, ex_model=DeepExModel, encode_mode=DeepExEncMode,
                                   max_memory_frames=DeepExMaxMemFrames,
                                   smooth=True, smooth_p=[0.3, 0.7, 0.9, 0.0, "none"], colormap=chroma_adjust)

        if ScMinFreq in range(1, 20):
            clip_colored = HAVC_stabilizer(clip_colored, stab_p=[5, 'A', 1, 15, 0.2, 0.15], colormap=chroma_adjust2)
        else:
            clip_colored = HAVC_stabilizer(clip_colored, stab_p=[3, 'A', 1, 0, 0, 0], colormap=chroma_adjust2)

    elif EnableDeepEx and DeepExMethod in (3, 4):

        clip_colored = HAVC_deepex(clip=clip, clip_ref=None, method=DeepExMethod, render_speed=DeepExPreset,
                                   render_vivid=DeepExVivid, sc_framedir=ScFrameDir,
                                   only_ref_frames=DeepExOnlyRefFrames,
                                   dark=True, dark_p=[0.2, 0.8], smooth=True, smooth_p=[0.3, 0.7, 0.9, 0.0, "none"],
                                   ex_model=DeepExModel, encode_mode=DeepExEncMode,
                                   max_memory_frames=DeepExMaxMemFrames, colormap=chroma_adjust)

    else:  # No DeepEx -> HAVC classic
        clip_colored = HAVC_ddeoldify(clip, method=dd_method, mweight=ddcolor_weight[w_id],
                                      deoldify_p=[0, deoldify_rf, 1.0, 0.0],
                                      ddcolor_p=[dd_model, ddcolor_rf, 1.0, 0.0, enable_fp16],
                                      ddtweak=dd_tweak, ddtweak_p=[0.0, 1.0, 2.5, True, 0.3, 0.6, 1.5, 0.5, hue_range])

        if pr_id > 5:  # 'faster', 'veryfast'
            clip_colored = HAVC_stabilizer(clip_colored, colormap=chroma_adjust)
        elif pr_id > 3:  # 'medium', 'fast' + 'faster', 'veryfast'
            clip_colored = HAVC_stabilizer(clip_colored, dark=True, dark_p=[0.2, 0.8], colormap=chroma_adjust,
                                           smooth=True, smooth_p=[0.3, 0.7, 0.9, 0.0, "none"],
                                           stab=True, stab_p=[5, 'A', 1, 15, 0.2, 0.15])
        else:  # 'placebo', 'veryslow', 'slower', 'slow'
            clip_colored = HAVC_stabilizer(clip_colored, dark=True, dark_p=[0.2, 0.8], colormap=chroma_adjust,
                                           smooth=True, smooth_p=[0.3, 0.7, 0.9, 0.0, "none"],
                                           stab=True, stab_p=[5, 'A', 1, 15, 0.2, 0.15, hue_range2])

    return clip_colored


"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
Exemplar-based coloring function with additional post-process filters 
"""


def HAVC_deepex(clip: vs.VideoNode = None, clip_ref: vs.VideoNode = None, method: int = 0, render_speed: str = 'medium',
                render_vivid: bool = True, ref_merge: int = 0, sc_framedir: str = None,
                only_ref_frames: bool = False, dark: bool = False, dark_p: list = (0.2, 0.8), smooth: bool = False,
                smooth_p: list = (0.3, 0.7, 0.9, 0.0, "none"), colormap: str = "none",
                ref_weight: float = None, ref_thresh: float = None, ex_model: int = 0, encode_mode: int = 0,
                max_memory_frames: int = 0, torch_dir: str = model_dir) -> vs.VideoNode:
    """Towards Video-Realistic Colorization via Exemplar-based framework

    :param clip:                Clip to process. Only RGB24 format is supported
    :param clip_ref:            Clip containing the reference frames (necessary if method=0,1,2,5)
    :param method:              Method to use to generate reference frames (RF).
                                        0 = HAVC same as video (default)
                                        1 = HAVC + RF same as video
                                        2 = HAVC + RF different from video
                                        3 = external RF same as video
                                        4 = external RF different from video
                                        5 = HAVC different from video
    :param render_speed:        Preset to control the render method and speed:
                                Allowed values are:
                                        'Fast'   (colors are more washed out)
                                        'Medium' (colors are a little washed out)
                                        'Slow'   (colors are a little more vivid)
    :param render_vivid:        Depending on selected ex_model, if enabled (True):
                                    0) ColorMNet: the frames memory is reset at every reference frame update
                                    1) Deep-Exemplar: the saturation will be increased by about 25%.
                                range [True, False]
    :param ref_merge:           Method used by DeepEx to merge the reference frames with the frames propagated by DeepEx.
                                It is applicable only with DeepEx method: 0, 1, 2, 5.
                                The HAVC reference frames must be produced with frequency = 1.
                                Allowed values are:
                                        0 = No RF merge (reference frames can be produced with any frequency)
                                        1 = RF-Merge VeryLow (reference frames are merged with weight=0.3)
                                        2 = RF-Merge Low (reference frames are merged with weight=0.4)
                                        3 = RF-Merge Med (reference frames are merged with weight=0.5)
                                        4 = RF-Merge High (reference frames are merged with weight=0.6)
                                        5 = RF-Merge VeryHigh (reference frames are merged with weight=0.7)
    :param ref_weight:          If (ref_merge > 0), represent the weight used to merge the reference frames.
                                If is not set, is assigned automatically a value depending on ref_merge value.
    :param ref_thresh:          If (ref_merge > 0), represent the threshold used to create the reference frames.
                                If is not set, is assigned automatically a value of 0.10
    :param sc_framedir:         If set, define the directory where are stored the reference frames. If only_ref_frames=True,
                                and method=0 this directory will be written with the reference frames used by the filter.
                                if method!=0 the directory will be read to create the reference frames that will be used
                                by "Exemplar-based" Video Colorization. The reference frame name must be in the
                                format: ref_nnnnnn.[jpg|png], for example the reference frame 897 must be
                                named: ref_000897.png
    :param only_ref_frames:     If enabled the filter will output in "sc_framedir" the reference frames. Useful to check
                                and eventually correct the frames with wrong colors.
    :param dark:                Enable/disable darkness filter (only on ref-frames), range [True,False]
    :param dark_p:              Parameters for darken the clip's dark portions, which sometimes are wrongly colored by the color models
                                      [0] : dark_threshold, luma threshold to select the dark area, range [0.1-0.5] (0.01=1%)
                                      [1] : dark_amount: amount of desaturation to apply to the dark area, range [0-1]
                                      [2] : "chroma range" parameter (optional), if="none" is disabled (see the README)
    :param smooth:              Enable/disable chroma smoothing (only on ref-frames), range [True, False]
    :param smooth_p:            parameters to adjust the saturation and "vibrancy" of the clip.
                                      [0] : dark_threshold, luma threshold to select the dark area, range [0-1] (0.01=1%)
                                      [1] : white_threshold, if > dark_threshold will be applied a gradient till white_threshold, range [0-1] (0.01=1%)
                                      [2] : dark_sat, amount of de-saturation to apply to the dark area, range [0-1]
                                      [3] : dark_bright, darkness parameter it used to reduce the "V" component in "HSV" colorspace, range [0, 1]
                                      [4] : "chroma range" parameter (optional), if="none" is disabled (see the README)
    :param colormap:            Direct hue/color mapping (only on ref-frames), without luma filtering, using the "chroma adjustment"
                                parameter, if="none" is disabled.
    :param ex_model:            "Exemplar-based" model to use for the color propagation, available models are:
                                    0 : ColorMNet (default)
                                    1 : Deep-Exemplar
    :param encode_mode:         Parameter used by ColorMNet to define the encode mode strategy.
                                Available values are:
                                     0: remote encoding. The frames will be colored by a thread outside Vapoursynth.
                                                         This option don't have any GPU memory limitation and will allow
                                                         to fully use the long term frame memory.
                                                         It is the faster encode method (default)
                                     1: local encoding.  The frames will be colored inside the Vapoursynth environment.
                                                         In this case the max_memory will be limited by the size of GPU
                                                         memory (max 15 frames for 24GB GPU).
                                                         Useful for coloring clips with a lot of smooth transitions,
                                                         since in this case is better to use a short frame memory or
                                                         the Deep-Exemplar model, which is faster.
    :param max_memory_frames:   Parameter used by ColorMNet model, specify the max number of encoded frames to keep in memory.
                                Its value depend on encode mode and must be defined manually following the suggested values.
                                encode_mode=0: there is no memory limit (it could be all the frames in the clip).
                                Suggested values are:
                                    min=150, max=10000
                                If = 0 will be filled with the value of 10000 or the clip length if lower.
                                encode_mode=1: the max memory frames is limited by available GPU memory.
                                Suggested values are:
                                    min=1, max=4    : for 8GB GPU
                                    min=1, max=8    : for 12GB GPU
                                    min=1, max=15   : for 24GB GPU
                                If = 0 will be filled with the max value (depending on total GPU RAM available)
    :param torch_dir:           torch hub dir location, default is model directory, if set to None will switch
                                to torch cache dir
    """
    # disable packages warnings
    disable_warnings()

    if not isinstance(clip, vs.VideoNode):
        HAVC_LogMessage(MessageType.EXCEPTION, "HAVC_deepex: this is not a clip")

    if not torch.cuda.is_available():
        HAVC_LogMessage(MessageType.EXCEPTION, "HAVC_deepex: CUDA is not available")

    if only_ref_frames and (sc_framedir is None):
        HAVC_LogMessage(MessageType.EXCEPTION, "HAVC_deepex: only_ref_frames is enabled but sc_framedir is unset")

    if not (sc_framedir is None) and method not in (0, 5) and only_ref_frames:
        HAVC_LogMessage(MessageType.EXCEPTION,
                        "HAVC_deepex: only_ref_frames is enabled but method not in (0, 5) (HAVC)")

    if method not in (0, 5) and (sc_framedir is None):
        HAVC_LogMessage(MessageType.EXCEPTION, "HAVC_deepex: method not in (0, 5) but sc_framedir is unset")

    if method in (3, 4) and not (clip_ref is None):
        HAVC_LogMessage(MessageType.EXCEPTION, "HAVC_deepex: method in (3, 4) but clip_ref is set")

    if method in (0, 1, 2, 5) and (clip_ref is None):
        HAVC_LogMessage(MessageType.EXCEPTION, "HAVC_deepex: method in (0, 1, 2, 5) but clip_ref is unset")

    # creates the directory "sc_framedir" and does not raise an exception if the directory already exists
    if not (sc_framedir is None):
        pathlib.Path(sc_framedir).mkdir(parents=True, exist_ok=True)

    if clip.format.id != vs.RGB24:
        HAVC_LogMessage(MessageType.WARNING, "HAVC_deepex: clip not in RGB24 format, it will be converted")
        # clip not in RGB24 format, it will be converted
        if clip.format.color_family == "YUV":
            clip = clip.resize.Bicubic(format=vs.RGB24, matrix_in_s="709", range_s="full",
                                       dither_type="error_diffusion")
        else:
            clip = clip.resize.Bicubic(format=vs.RGB24, range_s="full")

    if clip_ref is not None and clip_ref.format.id != vs.RGB24:
        HAVC_LogMessage(MessageType.WARNING, "HAVC_deepex: clip_ref not in RGB24 format, it will be converted")
        # clip_ref not in RGB24 format, it will be converted
        if clip_ref.format.color_family == "YUV":
            clip_ref = clip.resize.Bicubic(format=vs.RGB24, matrix_in_s="709", range_s="full",
                                           dither_type="error_diffusion")
        else:
            clip_ref = clip.resize.Bicubic(format=vs.RGB24, range_s="full")

    if method not in range(6):
        HAVC_LogMessage(MessageType.EXCEPTION, "HAVC_deepex: method must be in range [0-5]")

    if ref_merge not in range(6):
        HAVC_LogMessage(MessageType.EXCEPTION, "HAVC_deepex: ref_merge must be in range [0-5]")

    if ref_merge > 0 and method not in (0, 1):
        HAVC_LogMessage(MessageType.EXCEPTION,
                        "HAVC_deepex: method must be in range [0, 1] to be used with ref_merge > 0")

    if method in (0, 1, 2, 5):
        sc_threshold, sc_frequency = get_sc_props(clip_ref)
        if sc_threshold == 0 and sc_frequency == 0:
            HAVC_LogMessage(MessageType.EXCEPTION,
                            "HAVC_deepex: method in (0, 1, 2, 5) but sc_threshold and sc_frequency are not set")
        if sc_frequency == 1 and only_ref_frames:
            HAVC_LogMessage(MessageType.EXCEPTION,
                            "HAVC_deepex: only_ref_frames is enabled but sc_frequency == 1")
        if not only_ref_frames and ref_merge > 0 and sc_frequency != 1:
            HAVC_LogMessage(MessageType.EXCEPTION,
                            "HAVC_deepex: method in (0, 1, 2, 5) and ref_merge > 0 but sc_frequency != 1")

    if torch_dir is not None:
        torch.hub.set_dir(torch_dir)

    # static params
    enable_resize = False

    # unpack dark
    dark_enabled = dark
    dark_threshold = dark_p[0]
    dark_amount = dark_p[1]
    if len(dark_p) > 2:
        dark_hue_adjust = dark_p[2]
    else:
        dark_hue_adjust = 'none'

    # unpack chroma_smoothing
    chroma_smoothing_enabled = smooth
    black_threshold = smooth_p[0]
    white_threshold = smooth_p[1]
    dark_sat = smooth_p[2]
    dark_bright = -smooth_p[3]  # change the sign to reduce the bright
    if len(smooth_p) > 4:
        chroma_adjust = smooth_p[4]
    else:
        chroma_adjust = 'none'

    # define colormap
    colormap = colormap.lower()
    colormap_enabled = (colormap != "none" and colormap != "")

    enable_refmerge: bool = (ref_merge > 0 and sc_frequency == 1)
    refmerge_weight: list[float] = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7]
    if enable_refmerge:
        if ref_weight is None:
            ref_weight = refmerge_weight[ref_merge]
        if ref_thresh is None:
            ref_thresh = DEF_THRESHOLD
        clip_sc = SceneDetect(clip, threshold=ref_thresh)
        if method in (1, 2, 5) and not (sc_framedir is None) and not only_ref_frames:
            clip_sc = SceneDetectFromDir(clip_sc, sc_framedir=sc_framedir, merge_ref_frame=True,
                                         ref_frame_ext=(method in (2, 5)))
    else:
        ref_weight = 1.0
        clip_sc = None

    if method != 0 and not (sc_framedir is None):
        ref_frame_ext = method in (2, 4, 5)
        merge_ref_frame = method in (1, 2, 5)
        if method in (1, 2, 5):
            clip = SceneDetectFromDir(clip_ref, sc_framedir=sc_framedir, merge_ref_frame=merge_ref_frame,
                                      ref_frame_ext=ref_frame_ext)
            clip_ref = CopySCDetect(clip_ref, clip)
        else:
            clip = SceneDetectFromDir(clip, sc_framedir=sc_framedir, merge_ref_frame=merge_ref_frame,
                                      ref_frame_ext=ref_frame_ext)
    else:
        clip = CopySCDetect(clip, clip_ref)

    clip_orig = clip

    # if ex_model == 0 and render_speed.lower() == 'fast':
    #    render_speed = 'medium'

    d_size = get_deepex_size(render_speed=render_speed.lower(), enable_resize=enable_resize)
    smc = SmartResizeColorizer(d_size)
    smr = SmartResizeReference(d_size)

    if method != 0 and not (sc_framedir is None):
        if method in (1, 2, 5):
            clip_ref = vs_ext_reference_clip(clip_ref, sc_framedir=sc_framedir)
        else:
            clip_ref = vs_ext_reference_clip(clip, sc_framedir=sc_framedir)

    # clip and clip_ref are resized to match the frame size used for inference
    clip = smc.get_resized_clip(clip)
    clip_ref = smr.get_resized_clip(clip_ref)

    if colormap_enabled:
        clip_ref = vs_sc_colormap(clip_ref, colormap=colormap)

    if dark_enabled:
        clip_ref = vs_sc_dark_tweak(clip_ref, dark_threshold=dark_threshold, dark_amount=dark_amount,
                                    dark_hue_adjust=dark_hue_adjust.lower())

    if chroma_smoothing_enabled:
        clip_ref = vs_sc_chroma_bright_tweak(clip_ref, black_threshold=black_threshold, white_threshold=white_threshold,
                                             dark_sat=dark_sat, dark_bright=dark_bright,
                                             chroma_adjust=chroma_adjust.lower())
    ref_same_as_video = method == 3   # unico caso in cui è True il flag
    if only_ref_frames:
        clip_colored = clip_ref
    else:
        match ex_model:
            case 0:  # ColorMNet
                clip_colored = vs_colormnet(clip, clip_ref, clip_sc, image_size=-1, enable_resize=enable_resize,
                                            encode_mode=encode_mode, max_memory_frames=max_memory_frames,
                                            frame_propagate=ref_same_as_video, render_vivid=render_vivid,
                                            ref_weight=ref_weight)
            case 1:  # Deep-Exemplar
                clip_colored = vs_deepex(clip, clip_ref, clip_sc, image_size=d_size, enable_resize=enable_resize,
                                         propagate=ref_same_as_video, wls_filter_on=True, render_vivid=render_vivid,
                                         ref_weight=ref_weight)
            case _:
                HAVC_LogMessage(MessageType.EXCEPTION, "HybridAVC: unknown exemplar model id: " + str(ex_model))

    clip_resized = smc.restore_clip_size(clip_colored)

    # restore original resolution details, 5% faster than ShufflePlanes()
    if not (sc_framedir is None) and method in (0, 5) and only_ref_frames:
        # ref frames are saved if sc_framedir is set
        return vs_sc_recover_clip_luma(clip_orig, clip_resized, scenechange=True, sc_framedir=sc_framedir)
    else:
        return vs_recover_clip_luma(clip_orig, clip_resized)


"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
coloring function with additional pre-process and post-process filters 
"""


def HAVC_ddeoldify(
        clip: vs.VideoNode, method: int = 2, mweight: float = 0.4, deoldify_p: list = (0, 24, 1.0, 0.0),
        ddcolor_p: list = (1, 24, 1.0, 0.0, True), ddtweak: bool = False,
        ddtweak_p: list = (0.0, 1.0, 2.5, True, 0.3, 0.6, 1.5, 0.5, "300:360|0.8,0.1"),
        cmc_tresh: float = 0.2, lmm_p: list = (0.2, 0.8, 1.0), alm_p: list = (0.8, 1.0, 0.15), cmb_sw: bool = False,
        sc_threshold: float = 0.0, sc_tht_offset: int = 1, sc_min_freq: int = 0, sc_tht_ssim: float = 0.0,
        sc_normalize: bool = True, sc_min_int: int = 1, sc_tht_white: float = DEF_THT_WHITE,
        sc_tht_black: float = DEF_THT_BLACK, device_index: int = 0, torch_dir: str = model_dir,
        sc_debug: bool = False) -> vs.VideoNode:
    """A Deep Learning based project for colorizing and restoring old images and video using Deoldify and DDColor

    :param clip:                clip to process, only RGB24 format is supported
    :param method:              method used to combine deoldify() with ddcolor() (default = 2):
                                    0 : deoldify only (no merge)
                                    1 : ddcolor only (no merge)
                                    2 : Simple Merge (default):
                                        the frames are combined using a weighted merge, where the parameter "mweight"
                                        represent the weight assigned to the colors provided by the ddcolor() frames.
                                        With this method is suggested a starting weight < 50% (ex. = 40%).
                                    3 : Constrained Chroma Merge:
                                        given that the colors provided by deoldify() are more conservative and stable
                                        than the colors obtained with ddcolor(). The frames are combined by assigning
                                        a limit to the amount of difference in chroma values between deoldify() and
                                        ddcolor() this limit is defined by the threshold parameter "cmc_tresh".
                                        The limit is applied to the image converted to "YUV". For example when
                                        cmc_tresh=0.2, the chroma values "U","V" of ddcolor() frame will be constrained
                                        to have an absolute percentage difference respect to "U","V" provided by deoldify()
                                        not higher than 20%. The final limited frame will be merged again with the deoldify()
                                        frame. With this method is suggested a starting weight > 50% (ex. = 60%).
                                    4 : Luma Masked Merge:
                                        the frames are combined using a masked merge, the pixels of ddcolor() with luma < "luma_mask_limit"
                                        will be filled with the pixels of deoldify(). If "luma_white_limit" > "luma_mask_limit" the mask will
                                        apply a gradient till "luma_white_limit". If the parameter "mweight" > 0 the final masked frame will
                                        be merged again with the deoldify() frame. With this method is suggested a starting weight > 60%
                                        (ex. = 70%).
                                    5 : Adaptive Luma Merge:
                                        given that the ddcolor() perfomance is quite bad on dark scenes, the images are
                                        combined by decreasing the weight assigned to ddcolor() when the luma is
                                        below a given threshold given by: luma_threshold. The weight is calculated using
                                        the formula: merge_weight = max(mweight * (luma/luma_threshold)^alpha, min_weight).
                                        For example with: luma_threshold = 0.6 and alpha = 1, the weight assigned to
                                        ddcolor() will start to decrease linearly when the luma < 60% till "min_weight".
                                        For alpha=2, begins to decrease quadratically (because luma/luma_threshold < 1).
                                        With this method is suggested a starting weight > 70% (ex. = 80%).
                                    The methods 3 and 4 are similar to Simple Merge, but before the merge with deoldify()
                                    the ddcolor() frame is limited in the chroma changes (method 3) or limited based on the luma
                                    (method 4). The method 5 is a Simple Merge where the weight decrease with luma.
    :param mweight:             weight given to ddcolor's clip in all merge methods, range [0-1] (0.01=1%)
    :param deoldify_p:          parameters for deoldify():
                                   [0] deoldify model to use (default = 0):
                                       0 = ColorizeVideo_gen
                                       1 = ColorizeStable_gen
                                       2 = ColorizeArtistic_gen
                                   [1] render factor for the model, range: 10-44 (default = 24).
                                   [2] saturation parameter to apply to deoldify color model (default = 1)
                                   [3] hue parameter to apply to deoldify color model (default = 0)
    :param ddcolor_p:           parameters for ddcolor():
                                   [0] ddcolor model (default = 1):
                                       0 = ddcolor_modelscope,
                                       1 = ddcolor_artistic
                                       2 = colorization_siggraph17
                                       3 = colorization_eccv16
                                   [1] render factor for the model, if=0 will be auto selected
                                       (default = 24), range: [0, 10-64]
                                   [2] saturation parameter to apply to ddcolor color model (default = 1)
                                   [3] hue parameter to apply to ddcolor color model (default = 0)
                                   [4] enable/disable FP16 in ddcolor inference
    :param ddtweak:             enabled/disable tweak parameters for ddcolor(), range [True,False]
    :param ddtweak_p:           tweak parameters for ddcolor():
                                   [0] : ddcolor tweak's bright (default = 0)
                                   [1] : ddcolor tweak's contrast (default = 1), if < 1 ddcolor provides de-saturated frames
                                   [2] : ddcolor tweak's gamma (default = 1)
                                   [3] : luma constrained gamma -> luma constrained gamma correction enabled (default = False), range: [True, False]
                                            When enabled the average luma of a video clip will be forced to don't be below the value
                                            defined by the parameter "luma_min". The function allow to modify the gamma
                                            of the clip if the average luma is below the parameter "gamma_luma_min". A gamma value > 2.0 improves
                                            the ddcolor stability on bright scenes, while a gamma < 1 improves the ddcolor stability on
                                            dark scenes. The decrease of the gamma with luma is activated using a gamma_alpha != 0.
                                   [4] : luma_min: luma (%) min value for tweak activation (default = 0.2), if=0 is not activated, range [0-1]
                                   [5] : gamma_luma_min: luma (%) min value for gamma tweak activation (default = 0.5), if=0 is not activated, range [0-1]
                                   [6] : gamma_alpha: the gamma will decrease with the luma g = max(gamma * pow(luma/gamma_luma_min, gamma_alpha), gamma_min),
                                         for a movie with a lot of dark scenes is suggested alpha > 1, if=0 is not activated, range [>=0]
                                   [7] : gamma_min: minimum value for gamma, range (default=0.5) [>0.1]
                                   [8] : "chroma adjustment" parameter (optional), if="none" is disabled (see the README)
    :param cmc_tresh:           chroma_threshold (%), used by: "Constrained Chroma Merge", range [0-1] (0.01=1%)
    :param lmm_p:               parameters for method: "Luma Masked Merge" (see method=4 for a full explanation)
                                   [0] : luma_mask_limit: luma limit for build the mask used in Luma Masked Merge, range [0-1] (0.01=1%)
                                   [1] : luma_white_limit: the mask will apply a gradient till luma_white_limit, range [0-1] (0.01=1%)
                                   [2] : luma_mask_sat: if < 1 the ddcolor dark pixels will substitute with the desaturated deoldify pixels, range [0-1] (0.01=1%)
    :param alm_p:               parameters for method: "Adaptive Luma Merge" (see method=5 for a full explanation)
                                   [0] : luma_threshold: threshold for the gradient merge, range [0-1] (0.01=1%)
                                   [1] : alpha: exponent parameter used for the weight calculation, range [>0]
                                   [2] : min_weight: min merge weight, range [0-1] (0.01=1%)
    :param cmb_sw:              if true switch the clip order in all the combining methods, range [True,False]
    :param sc_threshold:        Scene change threshold used to generate the reference frames to be used by
                                "Exemplar-based" Video Colorization. It is a percentage of the luma change between
                                the previous and the current frame. range [0-1], default 0.0. If =0 are not generate
                                reference frames and will be colorized all the frames.
    :param sc_tht_offset:       Offset index used for the Scene change detection. The comparison will be performed,
                                between frame[n] and frame[n-offset]. An offset > 1 is useful to detect blended scene
                                change, range[1, 25]. Default = 1.
    :param sc_tht_ssim:         Threshold used by the SSIM (Structural Similarity Index Metric) selection filter.
                                If > 0, will be activated a filter that will improve the scene-change detection,
                                by discarding images that are similar.
                                Suggested values are between 0.35 and 0.85, range [0-1], default 0.0 (deactivated)
    :param sc_normalize:        If true the B&W frames are normalized before use misc.SCDetect(), the normalization will
                                increase the sensitivity to smooth scene changes.
    :param sc_min_int:          Minimum number of frame interval between scene changes, range[1, 25]. Default = 1.
    :param sc_min_freq:         if > 0 will be generate at least a reference frame every "sc_min_freq" frames.
                                range [0-1500], default: 0.
    :param sc_tht_white:        Threshold to identify white frames, range [0-1], default 0.85.
    :param sc_tht_black:        Threshold to identify dark frames, range [0-1], default 0.15.
    :param device_index:        device ordinal of the GPU, choices: GPU0...GPU7, CPU=99 (default = 0)
    :param torch_dir:           torch hub dir location, default is model directory, if set to None will switch to torch cache dir.
    :param sc_debug:            Print debug messages regarding the scene change detection process.
    """
    # disable packages warnings
    disable_warnings()

    if not torch.cuda.is_available() and device_index != 99:
        HAVC_LogMessage(MessageType.EXCEPTION, "HAVC_ddeoldify: CUDA is not available")

    if not isinstance(clip, vs.VideoNode):
        HAVC_LogMessage(MessageType.EXCEPTION, "HAVC_ddeoldify: this is not a clip")

    if sc_threshold < 0:
        HAVC_LogMessage(MessageType.EXCEPTION, "HAVC_ddeoldify: sc_threshold must be >= 0")

    if sc_min_freq < 0:
        HAVC_LogMessage(MessageType.EXCEPTION, "HAVC_ddeoldify: sc_min_freq must be >= 0")

    if method == 0:
        merge_weight = 0.0
    elif method == 1:
        merge_weight = 1.0
    else:
        merge_weight = mweight

    if merge_weight == 0.0:
        method = 0  # deoldify only
    elif merge_weight == 1.0:
        method = 1  # ddcolor only

    # unpack deoldify_params
    deoldify_model = deoldify_p[0]
    deoldify_rf = deoldify_p[1]
    deoldify_sat = deoldify_p[2]
    deoldify_hue = deoldify_p[3]

    # unpack deoldify_params
    ddcolor_model = ddcolor_p[0]
    ddcolor_rf = ddcolor_p[1]
    ddcolor_sat = ddcolor_p[2]
    ddcolor_hue = ddcolor_p[3]
    ddcolor_enable_fp16 = ddcolor_p[4]

    if os.path.getsize(os.path.join(model_dir, "ColorizeVideo_gen.pth")) == 0:
        HAVC_LogMessage(MessageType.EXCEPTION, "HAVC_ddeoldify: model files have not been downloaded.")

    if device_index > 7 and device_index != 99:
        HAVC_LogMessage(MessageType.EXCEPTION, "HAVC_ddeoldify: wrong device_index, choices are: GPU0...GPU7, CPU=99")

    if ddcolor_rf != 0 and ddcolor_rf not in range(10, 65):
        HAVC_LogMessage(MessageType.EXCEPTION, "HAVC_ddeoldify: ddcolor render_factor must be between: 10-64")

    if clip.format.id != vs.RGB24:
        # clip not in RGB24 format, it will be converted
        if clip.format.color_family == "YUV":
            clip = clip.resize.Bicubic(format=vs.RGB24, matrix_in_s="709", range_s="full",
                                       dither_type="error_diffusion")
        else:
            clip = clip.resize.Bicubic(format=vs.RGB24, range_s="full")

    # choices: GPU0...GPU7, CPU=99
    device.set(device=DeviceId(device_index))

    if torch_dir is not None:
        torch.hub.set_dir(torch_dir)

    if ddcolor_rf == 0:
        ddcolor_rf = min(max(math.trunc(0.4 * clip.width / 16), 16), 48)

    scenechange = not (sc_threshold == 0 and sc_min_freq == 0)

    clip = SceneDetect(clip, threshold=sc_threshold, frequency=sc_min_freq, sc_tht_filter=sc_tht_ssim,
                       tht_offset=sc_tht_offset, min_length=sc_min_int, frame_norm=sc_normalize,
                       tht_white=sc_tht_white, tht_black=sc_tht_black, sc_debug=sc_debug)

    frame_size = min(max(ddcolor_rf, deoldify_rf) * 16, clip.width)  # frame size calculation for inference()
    clip_orig = clip
    clip = clip.resize.Spline64(width=frame_size, height=frame_size)

    clipa = vs_sc_deoldify(clip, method=method, model=deoldify_model, render_factor=deoldify_rf,
                           scenechange=scenechange, package_dir=package_dir)
    clipb = vs_sc_ddcolor(clip, method=method, model=ddcolor_model, render_factor=ddcolor_rf, tweaks_enabled=ddtweak,
                          tweaks=ddtweak_p, enable_fp16=ddcolor_enable_fp16, scenechange=scenechange,
                          device_index=device_index)

    if scenechange:
        clip_colored = vs_sc_combine_models(clipa, clipb, method=method, sat=[deoldify_sat, ddcolor_sat],
                                            hue=[deoldify_hue, ddcolor_hue], clipb_weight=merge_weight,
                                            scenechange=True)
    else:
        clip_colored = vs_combine_models(clip_a=clipa, clip_b=clipb, method=method, sat=[deoldify_sat, ddcolor_sat],
                                         hue=[deoldify_hue, ddcolor_hue], clipb_weight=merge_weight, CMC_p=cmc_tresh,
                                         LMM_p=lmm_p, ALM_p=alm_p, invert_clips=cmb_sw)

    clip_resized = _clip_chroma_resize(clip_orig, clip_colored)
    return clip_resized


"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
Video color stabilization filter.
"""


def HAVC_stabilizer(clip: vs.VideoNode, dark: bool = False, dark_p: list = (0.2, 0.8), smooth: bool = False,
                    smooth_p: list = (0.3, 0.7, 0.9, 0.0, "none"), stab: bool = False,
                    stab_p: list = (5, 'A', 1, 15, 0.2, 0.15), colormap: str = "none",
                    render_factor: int = 24) -> vs.VideoNode:
    """Video color stabilization filter, which can be applied to stabilize the chroma components in colored clips.
        :param clip:                clip to process, only RGB24 format is supported.
        :param dark:                enable/disable darkness filter, range [True,False]
        :param dark_p:              parameters for darken the clip's dark portions, which sometimes are wrongly colored by the color models
                                      [0] : dark_threshold, luma threshold to select the dark area, range [0.1-0.5] (0.01=1%)
                                      [1] : dark_amount: amount of desaturation to apply to the dark area, range [0-1]
                                      [2] : "chroma range" parameter (optional), if="none" is disabled (see the README)
        :param smooth:              enable/disable chroma smoothing, range [True, False]
        :param smooth_p:            parameters to adjust the saturation and "vibrancy" of the clip.
                                      [0] : dark_threshold, luma threshold to select the dark area, range [0-1] (0.01=1%)
                                      [1] : white_threshold, if > dark_threshold will be applied a gradient till white_threshold, range [0-1] (0.01=1%)
                                      [2] : dark_sat, amount of de-saturation to apply to the dark area, range [0-1]
                                      [3] : dark_bright, darkness parameter it used to reduce the "V" component in "HSV" colorspace, range [0, 1]
                                      [4] : "chroma range" parameter (optional), if="none" is disabled (see the README)
        :param stab:                enable/disable chroma stabilizer, range [True, False]
        :param stab_p:              parameters for the temporal color stabilizer
                                      [0] : nframes, number of frames to be used in the stabilizer, range[3-15]
                                      [1] : mode, type of average used by the stabilizer: range['A'='arithmetic', 'W'='weighted']
                                      [2] : sat: saturation applied to the restored gray pixels [0,1]
                                      [3] : tht, threshold to detect gray pixels, range [0,255], if=0 is not applied the restore,
                                            its value depends on merge method used, suggested values are:
                                                method 0: tht = 5
                                                method 1: tht = 60 (ddcolor provides very saturated frames)
                                                method 2: tht = 15
                                                method 3: tht = 20
                                                method 4: tht = 5
                                                method 5: tht = 10
                                      [4] : weight, weight to blend the restored imaage (default=0.2), range [0-1], if=0 is not applied the blending
                                      [5] : tht_scen, threshold for scene change detection (default = 0.15), if=0 is not activated, range [0.01-0.50]
                                      [6] : "chroma adjustment" parameter (optional), if="none" is disabled (see the README)
        :param colormap:            direct hue/color mapping, without luma filtering, using the "chroma adjustment" parameter, if="none" is disabled
        :param render_factor:       render_factor to apply to the filters, the frame size will be reduced to speed-up the filters,
                                    but the final resolution will be the one of the original clip. If = 0 will be auto selected.
                                    This approach takes advantage of the fact that human eyes are much less sensitive to
                                    imperfections in chrominance compared to luminance. This means that it is possible to speed-up
                                    the chroma filters and and ultimately get a great high-resolution result, range: [0, 10-64]
    """

    if not isinstance(clip, vs.VideoNode):
        HAVC_LogMessage(MessageType.EXCEPTION, "HAVC_stabilizer: this is not a clip")

    if clip.format.id != vs.RGB24:
        # clip not in RGB24 format, it will be converted
        if clip.format.color_family == "YUV":
            clip = clip.resize.Bicubic(format=vs.RGB24, matrix_in_s="709", range_s="full",
                                       dither_type="error_diffusion")
        else:
            clip = clip.resize.Bicubic(format=vs.RGB24, range_s="full")

            # enable chroma_resize
    chroma_resize_enabled = True

    if render_factor != 0 and render_factor not in range(16, 65):
        HAVC_LogMessage(MessageType.EXCEPTION, "HAVC_stabilizer: render_factor must be between: 16-64")

    if render_factor == 0:
        render_factor = min(max(math.trunc(0.4 * clip.width / 16), 16), 64)

    if chroma_resize_enabled:
        frame_size = min(render_factor * 16, clip.width)  # frame size calculation for filters
        clip_orig = clip
        clip = clip.resize.Spline64(width=frame_size, height=frame_size)

    # unpack dark
    dark_enabled = dark
    dark_threshold = dark_p[0]
    dark_amount = dark_p[1]
    if len(dark_p) > 2:
        dark_hue_adjust = dark_p[2]
    else:
        dark_hue_adjust = 'none'

    # unpack chroma_smoothing
    chroma_smoothing_enabled = smooth
    black_threshold = smooth_p[0]
    white_threshold = smooth_p[1]
    dark_sat = smooth_p[2]
    dark_bright = -smooth_p[3]  # change the sign to reduce the bright
    if len(smooth_p) > 4:
        chroma_adjust = smooth_p[4]
    else:
        chroma_adjust = 'none'

    # define colormap
    colormap = colormap.lower()
    colormap_enabled = (colormap != "none" and colormap != "")

    # unpack chroma_stabilizer
    stab_enabled = stab
    stab_nframes = stab_p[0]
    stab_mode = stab_p[1]
    stab_sat = stab_p[2]
    stab_tht = stab_p[3]
    stab_weight = stab_p[4]
    stab_tht_scen = stab_p[5]
    if len(stab_p) > 6:
        stab_hue_adjust = stab_p[6]
    else:
        stab_hue_adjust = 'none'
    stab_algo = 0

    clip_colored = clip

    if dark_enabled:
        clip_colored = vs_dark_tweak(clip_colored, dark_threshold=dark_threshold, dark_amount=dark_amount,
                                     dark_hue_adjust=dark_hue_adjust.lower())

    if chroma_smoothing_enabled:
        clip_colored = vs_chroma_bright_tweak(clip_colored, black_threshold=black_threshold,
                                              white_threshold=white_threshold, dark_sat=dark_sat,
                                              dark_bright=dark_bright, chroma_adjust=chroma_adjust.lower())

    if colormap_enabled:
        clip_colored = vs_colormap(clip_colored, colormap=colormap)

    if stab_enabled:
        clip_colored = vs_chroma_stabilizer_ex(clip_colored, nframes=stab_nframes, mode=stab_mode, sat=stab_sat,
                                               tht=stab_tht, weight=stab_weight, hue_adjust=stab_hue_adjust.lower(),
                                               algo=stab_algo)

    if chroma_resize_enabled:
        return _clip_chroma_resize(clip_orig, clip_colored)
    else:
        return clip_colored


"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description: 
------------------------------------------------------------------------------- 
wrapper to function vSceneDetect() to set the scene-change frames in the clip
"""


def HAVC_SceneDetect(clip: vs.VideoNode, sc_threshold: float = DEF_THRESHOLD, sc_tht_offset: int = 1,
                     sc_tht_ssim: float = 0.0, sc_min_int: int = 1, sc_min_freq: int = 0, sc_normalize: bool = True,
                     sc_tht_white: float = DEF_THT_WHITE, sc_tht_black: float = DEF_THT_BLACK,
                     sc_debug: bool = False) -> vs.VideoNode:
    """Utility function to set the scene-change frames in the clip

    :param clip:                clip to process, only RGB24 format is supported.
    :param sc_threshold:        Scene change threshold used to generate the reference frames.
                                It is a percentage of the luma change between the previous n-frame (n=sc_tht_offset)
                                and the current frame. range [0-1], default 0.10.
    :param sc_tht_offset:       Offset index used for the Scene change detection. The comparison will be performed,
                                between frame[n] and frame[n-offset]. An offset > 1 is useful to detect blended scene
                                change, range[1, 25]. Default = 1.
    :param sc_normalize:        If true the B&W frames are normalized before use misc.SCDetect(), the normalization will
                                increase the sensitivity to smooth scene changes.
    :param sc_tht_white:        Threshold to identify white frames, range [0-1], default 0.85.
    :param sc_tht_black:        Threshold to identify dark frames, range [0-1], default 0.15.
    :param sc_tht_ssim:         Threshold used by the SSIM (Structural Similarity Index Metric) selection filter.
                                If > 0, will be activated a filter that will improve the scene-change detection,
                                by discarding images that are similar.
                                Suggested values are between 0.35 and 0.85, range [0-1], default 0.0 (deactivated)
    :param sc_min_int:          Minimum number of frame interval between scene changes, range[1, 25]. Default = 1.
    :param sc_min_freq:         if > 0 will be generated at least a reference frame every "sc_min_freq" frames.
                                range [0-1500], default: 0.
    :param sc_debug:            Enable SC debug messages. default: False

    """
    clip = SceneDetect(clip, threshold=sc_threshold, tht_offset=sc_tht_offset, frequency=sc_min_freq,
                       sc_tht_filter=sc_tht_ssim, min_length=sc_min_int, tht_white=sc_tht_white,
                       tht_black=sc_tht_black, frame_norm=sc_normalize, sc_debug=sc_debug)
    return clip


"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description: 
------------------------------------------------------------------------------- 
wrapper to function SceneDetect and vs_sc_export_frames() to export the clip's 
reference frames
"""


def HAVC_extract_reference_frames(clip: vs.VideoNode, sc_threshold: float = DEF_THRESHOLD, sc_tht_offset: int = 1,
                                  sc_tht_ssim: float = 0.0, sc_min_int: int = 1, sc_min_freq: int = 0,
                                  sc_framedir: str = "./", sc_normalize: bool = True, ref_offset: int = 0,
                                  sc_tht_white: float = DEF_THT_WHITE, sc_tht_black: float = DEF_THT_BLACK,
                                  ref_ext: str = "jpg", ref_jpg_quality: int = DEF_JPG_QUALITY,
                                  ref_override: bool = True, sc_debug: bool = False
                                  ) -> vs.VideoNode:
    """Utility function to export reference frames

    :param clip:                clip to process, only RGB24 format is supported.
    :param sc_threshold:        Scene change threshold used to generate the reference frames.
                                It is a percentage of the luma change between the previous n-frame (n=sc_offset)
                                and the current frame. range [0-1], default 0.05.
    :param sc_tht_offset:       Offset index used for the Scene change detection. The comparison will be performed,
                                between frame[n] and frame[n-offset]. An offset > 1 is useful to detect blended scene
                                change, range[1, 25]. Default = 1.
    :param sc_normalize:        If true the B&W frames are normalized before use misc.SCDetect(), the normalization will
                                increase the sensitivity to smooth scene changes.
    :param sc_tht_white:        Threshold to identify white frames, range [0-1], default 0.85.
    :param sc_tht_black:        Threshold to identify dark frames, range [0-1], default 0.15.
    :param sc_tht_ssim:         Threshold used by the SSIM (Structural Similarity Index Metric) selection filter.
                                If > 0, will be activated a filter that will improve the scene-change detection,
                                by discarding images that are similar.
                                Suggested values are between 0.35 and 0.85, range [0-1], default 0.0 (deactivated)
    :param sc_min_int:          Minimum number of frame interval between scene changes, range[1, 25]. Default = 1.
    :param sc_min_freq:         if > 0 will be generated at least a reference frame every "sc_min_freq" frames.
                                range [0-1500], default: 0.
    :param sc_framedir:         If set, define the directory where are stored the reference frames.
                                The reference frames are named as: ref_nnnnnn.[jpg|png].
    :param ref_offset:          Offset number that will be added to the number of generated frames. default: 0.
    :param ref_ext:             File extension and format of saved frames, range ["jpg", "png"] . default: "jpg"
    :param ref_jpg_quality:     Quality of "jpg" compression, range[0,100]. default: 95
    :param ref_override:        If True, the reference frames with the same name will be overridden, otherwise will
                                be discarded. default: True
    :param sc_debug:            Enable SC debug messages. default: False

    """
    pathlib.Path(sc_framedir).mkdir(parents=True, exist_ok=True)
    clip = SceneDetect(clip, threshold=sc_threshold, tht_offset=sc_tht_offset, frequency=sc_min_freq,
                       sc_tht_filter=sc_tht_ssim, min_length=sc_min_int, tht_white=sc_tht_white,
                       tht_black=sc_tht_black, frame_norm=sc_normalize, sc_debug=sc_debug)
    clip = vs_sc_export_frames(clip, sc_framedir=sc_framedir, ref_offset=ref_offset, ref_ext=ref_ext,
                               ref_jpg_quality=ref_jpg_quality, ref_override=ref_override)
    return clip


"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description: 
------------------------------------------------------------------------------- 
wrapper to function vs_sc_export_frames() to export the clip's reference frames
"""


def HAVC_export_reference_frames(clip: vs.VideoNode, sc_framedir: str = "./", ref_offset: int = 0,
                                 ref_ext: str = "jpg", ref_jpg_quality: int = DEF_JPG_QUALITY,
                                 ref_override: bool = True) -> vs.VideoNode:
    """Utility function to export reference frames

    :param clip:                clip to process, only RGB24 format is supported.
    :param sc_framedir:         If set, define the directory where are stored the reference frames.
                                The reference frames are named as: ref_nnnnnn.[jpg|png].
    :param ref_offset:          Offset number that will be added to the number of generated frames. default: 0.
    :param ref_ext:             File extension and format of saved frames, range ["jpg", "png"] . default: "jpg"
    :param ref_jpg_quality:     Quality of "jpg" compression, range[0,100]. default: 95
    :param ref_override:        If True, the reference frames with the same name will be overridden, otherwise will
                                be discarded. default: True
    """
    pathlib.Path(sc_framedir).mkdir(parents=True, exist_ok=True)
    clip = vs_sc_export_frames(clip, sc_framedir=sc_framedir, ref_offset=ref_offset, ref_ext=ref_ext,
                               ref_jpg_quality=ref_jpg_quality, ref_override=ref_override)
    return clip


"""
------------------------------------------------------------------------------------------------------------------------ 
                                   DDEOLDIFY LEGACY FUNCTIONS (deprecated)
------------------------------------------------------------------------------------------------------------------------ 
"""


def ddeoldify_main(clip: vs.VideoNode, Preset: str = 'Fast', VideoTune: str = 'Stable', ColorFix: str = 'Violet/Red',
                   ColorTune: str = 'Light', ColorMap: str = 'None', degrain_strength: int = 0,
                   enable_fp16: bool = True) -> vs.VideoNode:
    vs.core.log_message(vs.MESSAGE_TYPE_WARNING,
                        "Warning: ddeoldify_main is deprecated and may be removed in the future, please use 'HAVC_main' instead.")

    return HAVC_main(clip=clip, Preset=Preset, VideoTune=VideoTune, ColorFix=ColorFix, ColorTune=ColorTune,
                     ColorMap=ColorMap, enable_fp16=enable_fp16)


def ddeoldify(clip: vs.VideoNode, method: int = 2, mweight: float = 0.4, deoldify_p: list = (0, 24, 1.0, 0.0),
              ddcolor_p: list = (1, 24, 1.0, 0.0, True),
              dotweak: bool = False, dotweak_p: list = (0.0, 1.0, 1.0, False, 0.2, 0.5, 1.5, 0.5),
              ddtweak: bool = False, ddtweak_p: list = (0.0, 1.0, 2.5, True, 0.3, 0.6, 1.5, 0.5, "300:360|0.8,0.1"),
              degrain_strength: int = 0, cmc_tresh: float = 0.2, lmm_p: list = (0.2, 0.8, 1.0),
              alm_p: list = (0.8, 1.0, 0.15), cmb_sw: bool = False, device_index: int = 0,
              torch_dir: str = model_dir) -> vs.VideoNode:
    vs.core.log_message(vs.MESSAGE_TYPE_WARNING,
                        "Warning: ddeoldify is deprecated and may be removed in the future, please use 'HAVC_ddeoldify' instead.")

    return HAVC_ddeoldify(clip, method, mweight, deoldify_p, ddcolor_p, ddtweak, ddtweak_p, cmc_tresh, lmm_p, alm_p,
                          cmb_sw,
                          sc_threshold=0, sc_min_freq=0, device_index=device_index, torch_dir=torch_dir)


def ddeoldify_stabilizer(clip: vs.VideoNode, dark: bool = False, dark_p: list = (0.2, 0.8), smooth: bool = False,
                         smooth_p: list = (0.3, 0.7, 0.9, 0.0, "none"),
                         stab: bool = False, stab_p: list = (5, 'A', 1, 15, 0.2, 0.15), colormap: str = "none",
                         render_factor: int = 24) -> vs.VideoNode:
    vs.core.log_message(vs.MESSAGE_TYPE_WARNING,
                        "Warning: ddeoldify_stabilizer is deprecated and may be removed in the future, please use 'HAVC_stabilizer' instead.")

    return HAVC_stabilizer(clip, dark, dark_p, smooth, smooth_p, stab, stab_p, colormap, render_factor)


"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description: ONLY FOR TESTING
------------------------------------------------------------------------------- 
wrapper to function vs_sc_export_frames() to export the clip's reference frames
"""


def _extract_reference_frames(clip: vs.VideoNode, sc_framedir: str = "./", ref_offset: int = 0, ref_ext: str = "png",
                              ref_override: bool = True, prop_name: str = "_SceneChangePrev") -> vs.VideoNode:
    pathlib.Path(sc_framedir).mkdir(parents=True, exist_ok=True)
    clip = vs_sc_export_frames(clip, sc_framedir=sc_framedir, ref_offset=ref_offset, ref_ext=ref_ext,
                               ref_override=ref_override, prop_name=prop_name)
    return clip


"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description: ONLY FOR TESTING
------------------------------------------------------------------------------- 
wrapper to function vs_recover_clip_luma().
"""


def _clip_chroma_resize(clip_hires: vs.VideoNode, clip_lowres: vs.VideoNode) -> vs.VideoNode:
    clip_resized = clip_lowres.resize.Spline64(width=clip_hires.width, height=clip_hires.height)
    return vs_recover_clip_luma(clip_hires, clip_resized)


"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description: ONLY FOR TESTING
------------------------------------------------------------------------------- 
wrapper to function vs_get_clip_frame() to get frames fast.
"""


def _get_clip_frame(clip: vs.VideoNode, nframe: int = 0) -> vs.VideoNode:
    clip = vs_get_clip_frame(clip=clip, nframe=nframe)
    return clip


"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description: ONLY FOR TESTING
------------------------------------------------------------------------------- 
wrapper to function vs_recover_clip_color() to restore gray frames.
"""


def _recover_clip_color(clip: vs.VideoNode = None, clip_color: vs.VideoNode = None, sat: float = 1.0, tht: int = 0,
                        weight: float = 0.2, tht_scen: float = 0.8, hue_adjust: str = 'none',
                        return_mask: bool = False) -> vs.VideoNode:
    clip = vs_recover_clip_color(clip=clip, clip_color=clip_color, sat=sat, tht=tht, weight=weight, tht_scen=tht_scen,
                                 hue_adjust=hue_adjust, return_mask=return_mask)
    return clip


"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description: ONLY FOR TESTING
------------------------------------------------------------------------------- 
disable packages warnings.
"""


def disable_warnings():
    logger_blocklist = [
        "matplotlib",
        "PIL",
        "torch",
        "numpy",
        "tensorrt",
        "torch_tensorrt"
        "kornia",
        "dinov2"  # dinov2 is issuing warnings not allowing ColorMNetServer to work properly
    ]

    for module in logger_blocklist:
        logging.getLogger(module).setLevel(logging.ERROR)

    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=UserWarning)
    warnings.simplefilter(action='ignore', category=DeprecationWarning)
    # warnings.simplefilter(action="ignore", category=Warning)

    torch._logging.set_logs(all=logging.ERROR)
