"""
------------------------------------------------------------------------------- 
Author: Dan64
Date: 2024-02-29
version: 
LastEditors: Dan64
LastEditTime: 2025-02-21
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
Deep-Remaster: https://github.com/satoshiiizuka/siggraphasia2019_remastering
"""
from __future__ import annotations
from functools import partial

import os
import pathlib

import vsdeoldify.remaster

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
from vsdeoldify.havc_utils import *

__version__ = "5.0.4"

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


def HAVC_main(clip: vs.VideoNode, Preset: str = 'Medium', ColorModel: str = 'Video+Artistic', CombMethod: str = 'Simple',
              VideoTune: str = 'Stable', ColorFix: str = 'Violet/Red', ColorTune: str = 'Light', ColorMap: str = 'None',
              BlackWhiteTune: str = 'None', EnableDeepEx: bool = False, DeepExMethod: int = 0,
              DeepExPreset: str = 'Medium', DeepExRefMerge: int = 0, DeepExOnlyRefFrames: bool = False,
              ScFrameDir: str = None, ScThreshold: float = DEF_THRESHOLD, ScThtOffset: int = 1, ScMinFreq: int = 0,
              ScMinInt: int = 1, ScThtSSIM: float = 0.0, ScNormalize: bool = False, DeepExModel: int = 0,
              DeepExVivid: bool = True, DeepExEncMode: int = 0, DeepExMaxMemFrames=0,
              RefRange: tuple[int, int] = (0, 0),
              enable_fp16: bool = True, sc_debug: bool = False) -> vs.VideoNode:
    """Main HAVC function supporting the Presets

    :param clip:                clip to process, only RGB24 format is supported.
    :param Preset:              Preset to control the encoding speed/quality.
                                Allowed values are:
                                    'Placebo',
                                    'VerySlow',
                                    'Slower',
                                    'Slow',
                                    'Medium', (default)
                                    'Fast',  
                                    'Faster',
                                    'VeryFast'
    :param ColorModel:          Preset to control the Color Models to be used for the color inference
                                Allowed values are:
                                    'Video+Artistic'  (default)
                                    'Stable+Artistic'
                                    'Video+ModelScope'
                                    'Stable+ModelScope'
                                    'Artistic+Modelscope'
                                    'Video+Siggraph17'
                                    'Video+ECCV16'
                                    'DeOldify(Video)'
                                    'DeOldify(Stable)'
                                    'DeOldify(Artistic)'
                                    'DDColor(Artistic)'
                                    'DDColor(ModelScope)'
                                    'Zhang(Siggraph17)'
                                    'Zhang(ECCV16)'
    :param CombMethod:          Method used to combine coloring models with (+):
                                Allowed values are:
                                    'Simple' (default)
                                    'Constrained-Chroma'
                                    'Luma-Masked'
                                    'Adaptive-Luma'
    :param VideoTune:           Preset to control the output video color stability
                                Allowed values are:
                                    'VeryStable',
                                    'MoreStable'
                                    'Stable',  (default)
                                    'Balanced',
                                    'Vivid',
                                    'MoreVivid',
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
                                    'None'
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
    :param BlackWhiteTune:      This parameter allows to improve contrast and luminosity of Black & White input clip to
                                be colored with HAVC. Allowed values are:
                                    'None' (default)
                                    'Light',
                                    'Medium',
                                    'Strong'
    :param EnableDeepEx:        Enable coloring using "Exemplar-based" Video Colorization models
    :param DeepExMethod:        Method to use to generate reference frames.
                                        0 = HAVC same as video (default)
                                        1 = HAVC + RF same as video
                                        2 = HAVC + RF different from video
                                        3 = external RF same as video
                                        4 = external RF different from video
                                        5 = external ClipRef same as video
                                        6 = external ClipRef different from video
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
                                (can be used only if DeepExMethod = 0)
    :param DeepExModel:         Exemplar Model used by DeepEx to propagate color frames.
                                        0 : ColorMNet (default)
                                        1 : Deep-Exemplar
                                        2 : Deep-Remaster
    :param DeepExVivid:         Depending on selected DeepExModel, if enabled (True):
                                    0) ColorMNet: the frames memory is reset at every reference frame update
                                    1) Deep-Exemplar: the saturation will be increased by about 25%.
                                    2) Deep-Remaster: the saturation will be increased by about 20% and Hue by +10.
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
                                     2: remote all-ref   Same as "remote encoding" but all the available reference frames
                                                         will be used for the inference at the beginning of encoding.
    :param DeepExMaxMemFrames:  Parameter used by ColorMNet/DeepRemaster models.
                                For ColorMNet specify the max number of encoded frames to keep in memory.
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
                                For DeepRemaster represent the number to reference frames to keep in memory.
                                Suggested values are:
                                    min=4, max=50
                                If = 0 will be filled with the value of 20.
    :param ScFrameDir:          if set, define the directory where are stored the reference frames that will be used
                                by "Exemplar-based" Video Colorization models. With DeepExMethod 5,6 this parameter 
                                can be the path to a video clip.
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
                                increase the sensitivity to smooth scene changes, range [True, False], default: False
    :param RefRange:            Parameter used only with DeepExMethod in (5, 6). With this parameter it is possible to
                                provide the frame number of clip start and end. For example RefRange=(100, 500)
                                will return the clip's slice: clip[100:500], if RefRange=(0, 0) will be considered all
                                clip's frames.
    :param enable_fp16:         Enable/disable FP16 in ddcolor inference, range [True, False]
    :param sc_debug:            Print debug messages regarding the scene change detection process.
    """
    # disable packages warnings
    disable_warnings()

    # Select presets / tuning
    speed_id, deoldify_rf, ddcolor_rf = havc_utils._get_render_factors(Preset)

    # Select VideoTune
    ddcolor_weight = havc_utils._get_mweight(VideoTune)

    # Select Color model
    do_model, dd_model, dd_method = havc_utils._get_color_model(ColorModel)

    if dd_method == 2:
        dd_method = havc_utils._get_comb_method(CombMethod)

    dd_tweak, hue_range, hue_range2, chroma_adjust, chroma_adjust2 = havc_utils._get_color_tune(ColorTune, ColorFix,
                                                                                                ColorMap, dd_model)

    # ---------------------- START COLORING ------------------------------------
    clip = HAVC_bw_tune(clip, BlackWhiteTune, action='on')

    if EnableDeepEx and DeepExMethod in (0, 1, 2, 5, 6):

        havc_utils._check_input(DeepExOnlyRefFrames, ScFrameDir, DeepExMethod, ScThreshold, ScMinFreq, DeepExRefMerge)

        if ScMinFreq > 1:
            ref_freq = ScMinFreq
        else:
            ref_freq = 0

        if DeepExRefMerge > 0:
            ScMinFreq = 1

        if ScThreshold is not None and 0 < ScThreshold < 1:
            ref_tresh = ScThreshold
        else:
            ref_tresh = DEF_THRESHOLD

        if DeepExMethod in (5, 6):
            clip_ref = HAVC_read_video(source=ScFrameDir, fpsnum=clip.fps_num, fpsden=clip.fps_den)

            clip_s: int = RefRange[0]
            clip_e: int = RefRange[1]

            if clip_e > 0 and 0 <= clip_s <= clip_e:
                clip_ref = clip_ref[clip_s: clip_e]

            clip_colored = HAVC_restore_video(clip, clip_ref, render_speed=DeepExPreset, ex_model=DeepExModel,
                                              ref_merge=DeepExRefMerge, ref_thresh=ref_tresh, ref_freq=ref_freq,
                                              max_memory_frames=DeepExMaxMemFrames, render_vivid=DeepExVivid,
                                              encode_mode=DeepExEncMode, ref_norm=ScNormalize)

        else:
            clip_ref = HAVC_colorizer(clip, method=dd_method, mweight=ddcolor_weight,
                                      deoldify_p=[do_model, deoldify_rf, 1.0, 0.0],
                                      ddcolor_p=[dd_model, ddcolor_rf, 1.0, 0.0, enable_fp16],
                                      ddtweak=dd_tweak, ddtweak_p=[0.0, 1.0, 2.5, True, 0.3, 0.6, 1.5, 0.5, hue_range],
                                      sc_threshold=ScThreshold, sc_tht_offset=ScThtOffset, sc_min_freq=ScMinFreq,
                                      sc_min_int=ScMinInt, sc_tht_ssim=ScThtSSIM, sc_normalize=ScNormalize,
                                      sc_debug=sc_debug)
            clip_colored = HAVC_deepex(clip=clip, clip_ref=clip_ref, method=DeepExMethod, render_speed=DeepExPreset,
                                       render_vivid=DeepExVivid, ref_merge=DeepExRefMerge, sc_framedir=ScFrameDir,
                                       only_ref_frames=DeepExOnlyRefFrames, dark=True, dark_p=[0.2, 0.8],
                                       ref_thresh=ref_tresh, ex_model=DeepExModel, encode_mode=DeepExEncMode,
                                       max_memory_frames=DeepExMaxMemFrames, ref_freq=ref_freq, ref_norm=ScNormalize,
                                       smooth=True, smooth_p=[0.3, 0.7, 0.9, 0.0, "none"], colormap=chroma_adjust)

        # depending on the frequency of reference frames are adopted the faster stabilization settings
        if ScMinFreq in range(1, 20):
            clip_colored = HAVC_stabilizer(clip_colored, stab_p=[5, 'A', 1, 15, 0.2, 0.8], colormap=chroma_adjust2)
        else:
            clip_colored = HAVC_stabilizer(clip_colored, stab_p=[3, 'A', 1, 0, 0, 0], colormap=chroma_adjust2)

    elif EnableDeepEx and DeepExMethod in (3, 4):

        if DeepExModel == 2:
            # call to faster version of DeepRemaster that read directly the images folder (mode=0)
            clip_colored = HAVC_DeepRemaster(clip, render_vivid=DeepExVivid, ref_dir=ScFrameDir,
                                             ref_buffer_size=DeepExMaxMemFrames, mode=0)

        else:

            clip_colored = HAVC_deepex(clip=clip, clip_ref=None, method=DeepExMethod, render_speed=DeepExPreset,
                                       render_vivid=DeepExVivid, sc_framedir=ScFrameDir,
                                       only_ref_frames=DeepExOnlyRefFrames, dark=True, dark_p=[0.2, 0.8],
                                       smooth=True, smooth_p=[0.3, 0.7, 0.9, 0.0, "none"], ex_model=DeepExModel,
                                       encode_mode=DeepExEncMode, max_memory_frames=DeepExMaxMemFrames,
                                       colormap=chroma_adjust)

    else:  # No DeepEx -> HAVC classic
        clip_colored = HAVC_colorizer(clip, method=dd_method, mweight=ddcolor_weight,
                                      deoldify_p=[do_model, deoldify_rf, 1.0, 0.0],
                                      ddcolor_p=[dd_model, ddcolor_rf, 1.0, 0.0, enable_fp16],
                                      ddtweak=dd_tweak, ddtweak_p=[0.0, 1.0, 2.5, True, 0.3, 0.6, 1.5, 0.5, hue_range])

        if speed_id > 5:  # 'faster', 'veryfast' -> is used only colormap
            clip_colored = HAVC_stabilizer(clip_colored, colormap=chroma_adjust)
        elif speed_id > 3:  # 'medium', 'fast' -> are used all the filters except hue_range2
            clip_colored = HAVC_stabilizer(clip_colored, dark=True, dark_p=[0.2, 0.8], colormap=chroma_adjust,
                                           smooth=True, smooth_p=[0.3, 0.7, 0.9, 0.0, "none"],
                                           stab=True, stab_p=[5, 'A', 1, 15, 0.2, 0.8])
        else:  # 'placebo', 'veryslow', 'slower', 'slow' -> are used all the filters
            clip_colored = HAVC_stabilizer(clip_colored, dark=True, dark_p=[0.2, 0.8], colormap=chroma_adjust,
                                           smooth=True, smooth_p=[0.3, 0.7, 0.9, 0.0, "none"],
                                           stab=True, stab_p=[5, 'A', 1, 15, 0.2, 0.8, hue_range2])

    clip_colored = HAVC_bw_tune(clip_colored, BlackWhiteTune, action='off')

    return clip_colored


"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
Pre/post - process  filter for improving contrast and luminosity of clips to be
colored with HAVC.
"""


def HAVC_bw_tune(clip: vs.VideoNode = None, bw_tune: str = 'none', action: str = None,
                 hue: float = 0, sat: float = 1, bright: float = 0, cont: float = 1) -> vs.VideoNode:
    """Pre/post - process filter for improving contrast and luminosity of B&W/Colored clips

        :param clip:        Clip to process. Only RGB24 format is supported.
        :param bw_tune:     This parameter allows to improve contrast and luminosity of input clip to
                            be colored with HAVC. Allowed values are:
                                 'None' (default)
                                 'Light',
                                 'Medium',
                                 'Strong'
                                 'Custom' (allow to adjust: hue, saturation, brightness, contrast)
        :param action:      This parameter allows to apply the improvement to Black & White clips and to revert
                            the adjustments.
                            Allowed values are:
                                 'ON': the adjustments are applied on the input clip
                                 'OFF', the adjustments previous applied are almost reverted
                                 None, allowed only for bw_tune = 'Custom'
        :param hue:         Adjust the color hue of the image.
                                 hue>0.0 shifts the image towards red.
                                 hue<0.0 shifts the image towards green.
                            Range -180.0 to +180.0, default = 0.0
        :param sat:         Adjust the color saturation of the image by controlling gain of the color channels.
                                 sat>1.0 increases the saturation.
                                 sat<1.0 reduces the saturation.
                            Use sat=0 to convert to GreyScale.
                            Range 0.0 to 10.0, default = 1.0
        :param bright:      Change the brightness of the image by applying a constant bias to the luma channel.
                                 bright>0.0 increases the brightness.
                                 bright<0.0 decreases the brightness.
                            Range -255.0 to 255.0, default = 0.0
        :param cont:        Change the contrast of the image by multiplying the luma values by a constant.
                                 cont>1.0 increase the contrast (the luma range will be stretched).
                                 cont<1.0 decrease the contrast (the luma range will be contracted).
                            Range 0.0 to 10.0, default = 1.0

    """
    bw_tune_p = bw_tune.lower()
    bw_tune = ['none', 'light', 'medium', 'strong', 'custom']
    bw_cont_on = [1.0, 0.95, 0.90, 0.80, 1.0]
    bw_cont_off = [1.0, 1.03, 1.08, 1.15, 1.0]
    bw_bright_on = [0.0, -1, -2, -4, 0.0]
    bw_bright_off = [0.0, +1, +2, +4, 0.0]

    bw_id = 0
    try:
        bw_id = bw_tune.index(bw_tune_p)
    except ValueError:
        HAVC_LogMessage(MessageType.EXCEPTION, "HAVC_bw_tune: B&W tune choice is invalid: ", bw_tune_p)

    if bw_id == 4:
        return vs_tweak(clip, hue=hue, sat=sat, bright=bright, cont=cont)

    if bw_id == 0 or action is None:
        return clip

    action = action.lower()

    if action == 'on':
        clip_new = vs_tweak(clip, cont=bw_cont_on[bw_id], bright=bw_bright_on[bw_id])
    elif action == 'off':
        clip_new = vs_tweak(clip, cont=bw_cont_off[bw_id], bright=bw_bright_off[bw_id])
    else:
        HAVC_LogMessage(MessageType.EXCEPTION, "HAVC_bw_tune: B&W tune invalid action: ", action)

    return clip_new


"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
Exemplar-based coloring function with additional post-process filters 
"""


def HAVC_deepex(clip: vs.VideoNode = None, clip_ref: vs.VideoNode = None, method: int = 0, render_speed: str = 'medium',
                render_vivid: bool = True, ref_merge: int = 0, sc_framedir: str = None, ref_norm: bool = False,
                only_ref_frames: bool = False, dark: bool = False, dark_p: list = (0.2, 0.8), smooth: bool = False,
                smooth_p: list = (0.3, 0.7, 0.9, 0.0, "none"), colormap: str = "none", ref_weight: float = None,
                ref_thresh: float = None, ref_freq: int = None, ex_model: int = 0, encode_mode: int = 0,
                max_memory_frames: int = 0, torch_dir: str = model_dir) -> vs.VideoNode:
    """Towards Video-Realistic Colorization via Exemplar-based framework

    :param clip:                Clip to process. Only RGB24 format is supported
    :param clip_ref:            Clip containing the reference frames (necessary if method=0,1,2,5,6)
    :param method:              Method to use to generate reference frames (RF).
                                        0 = HAVC same as video (default)
                                        1 = HAVC + RF same as video
                                        2 = HAVC + RF different from video
                                        3 = external RF same as video
                                        4 = external RF different from video
                                        5 = external ClipRef same as video
                                        6 = external ClipRef different from video
    :param render_speed:        Preset to control the render method and speed:
                                Allowed values are:
                                        'Fast'   (colors are more washed out)
                                        'Medium' (colors are a little washed out)
                                        'Slow'   (colors are a little more vivid)
    :param render_vivid:        Depending on selected ex_model, if enabled (True):
                                    0) ColorMNet: the frames memory is reset at every reference frame update
                                    1) Deep-Exemplar: the saturation will be increased by about 25%.
                                    2) Deep-Remaster: the saturation will be increased by about 15%.
                                range [True, False]
    :param ref_merge:           Method used by DeepEx to merge the reference frames with the frames propagated by DeepEx.
                                It is applicable only with DeepEx method: 0, 1, 5.
                                The HAVC reference frames must be produced with frequency = 1.
                                Allowed values are:
                                        0 = No RF merge (reference frames can be produced with any frequency)
                                        1 = RF-Merge VeryLow (reference frames are merged with weight=0.3)
                                        2 = RF-Merge Low (reference frames are merged with weight=0.4)
                                        3 = RF-Merge Med (reference frames are merged with weight=0.5)
                                        4 = RF-Merge High (reference frames are merged with weight=0.6)
                                        5 = RF-Merge VeryHigh (reference frames are merged with weight=0.7)
    :param ref_weight:          If (ref_merge > 0), represent the weight used to merge the reference frames.
                                If is not set, is assigned automatically a value depending on ref_merge/method values.
    :param ref_thresh:          Represent the threshold used to create the reference frames. If is not set, is assigned
                                automatically a value of 0.10
    :param ref_freq:            If > 0 will be generated at least a reference frame every "ref_freq" frames.
                                range [0-1500]. If is not set, is assigned automatically a value depending on
                                ref_merge/method values.
    :param ref_norm:            If true the B&W frames are normalized before apply the Scene Detection to generate the
                                reference frames. The normalization will increase the sensitivity to smooth scene changes,
                                range [True, False], default: False
    :param sc_framedir:         If set, define the directory where are stored the reference frames. If only_ref_frames=True,
                                and method=0 this directory will be written with the reference frames used by the filter.
                                if method!=0 the directory will be read to create the reference frames that will be used
                                by "Exemplar-based" Video Colorization. The reference frame name must be in the
                                format: ref_nnnnnn.[jpg|png], for example the reference frame 897 must be
                                named: ref_000897.png. With methods 5,6 this parameters can be the path to a video clip.
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
                                    2 : Deep-Remaster
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
                                     2: remote all-ref   Same as "remote encoding" but all the available reference frames
                                                         will be used for the inference at the beginning of encoding.
    :param max_memory_frames:   Parameter used by ColorMNet/DeepRemaster models.
                                For ColorMNet specify the max number of encoded frames to keep in memory.
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
                                If = 0 will be filled with the max value (depending on total GPU RAM available).
                                For DeepRemaster represent the number to reference frames to keep in memory.
                                Suggested values are:
                                    min=2, max=50
                                If = 0 will be filled with the value of 20.
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

    if not (sc_framedir is None) and method != 0 and only_ref_frames:
        HAVC_LogMessage(MessageType.EXCEPTION,
                        "HAVC_deepex: only_ref_frames is enabled but method != 0 (HAVC)")

    if method != 0 and (sc_framedir is None):
        HAVC_LogMessage(MessageType.EXCEPTION, "HAVC_deepex: method != 0 but sc_framedir is unset")

    if method in (3, 4) and not (clip_ref is None):
        HAVC_LogMessage(MessageType.EXCEPTION, "HAVC_deepex: method in (3, 4) but clip_ref is set")

    if method in (0, 1, 2, 5, 6) and (clip_ref is None):
        HAVC_LogMessage(MessageType.EXCEPTION, "HAVC_deepex: method in (0, 1, 2, 5, 6) but clip_ref is unset")

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

    if method not in range(7):
        HAVC_LogMessage(MessageType.EXCEPTION, "HAVC_deepex: method must be in range [0-6]")

    if ref_merge not in range(6):
        HAVC_LogMessage(MessageType.EXCEPTION, "HAVC_deepex: ref_merge must be in range [0-5]")

    if ref_merge > 0 and method not in (0, 1, 5):
        HAVC_LogMessage(MessageType.EXCEPTION,
                        "HAVC_deepex: method must be in (0, 1, 5) to be used with ref_merge > 0")

    sc_threshold = None
    sc_frequency = None
    if method in (0, 1, 2):
        sc_threshold, sc_frequency = get_sc_props(clip_ref)
        if sc_threshold == 0 and sc_frequency == 0:
            HAVC_LogMessage(MessageType.EXCEPTION,
                            "HAVC_deepex: method in (0, 1, 2) but sc_threshold and sc_frequency are not set")
        if sc_frequency == 1 and only_ref_frames:
            HAVC_LogMessage(MessageType.EXCEPTION,
                            "HAVC_deepex: only_ref_frames is enabled but sc_frequency == 1")
        if not only_ref_frames and ref_merge > 0 and sc_frequency != 1:
            HAVC_LogMessage(MessageType.EXCEPTION,
                            "HAVC_deepex: method in (0, 1, 2) and ref_merge > 0 but sc_frequency != 1")

    if torch_dir is not None:
        torch.hub.set_dir(torch_dir)

    # ------------------------------------------------------------------------------------------------------------------
    # SPECIAL MANAGEMENT OF DeepRemaster (ex_model = 2)
    if method in (0, 1, 2) and ex_model == 2:
        HAVC_LogMessage(MessageType.EXCEPTION,
                        "HAVC_deepex: DeepRemaster cannot be used with methods: 0, 1, 2 (HAVC)")

    # SPECIAL MANAGEMENT OF METHOD=(5,6)
    if method in (5, 6):
        return HAVC_restore_video(clip, clip_ref, method, render_speed, ex_model, ref_merge, ref_weight, ref_thresh,
                                  ref_freq, ref_norm, max_memory_frames, render_vivid, encode_mode)
    # ------------------------------------------------------------------------------------------------------------------

    # creates the directory "sc_framedir" and does not raise an exception if the directory already exists
    if not (sc_framedir is None):
        pathlib.Path(sc_framedir).mkdir(parents=True, exist_ok=True)

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
        if ref_freq is None or ref_freq == 1:
            ref_freq = 0
        clip_sc = SceneDetect(clip, threshold=ref_thresh, frequency=ref_freq, frame_norm=ref_norm)
        if method in (1, 2) and not (sc_framedir is None) and not only_ref_frames:
            clip_sc = SceneDetectFromDir(clip_sc, sc_framedir=sc_framedir, merge_ref_frame=True,
                                         ref_frame_ext=(method == 2))
    else:
        ref_weight = 1.0
        clip_sc = None

    if method != 0 and not (sc_framedir is None):
        ref_frame_ext = method in (2, 4)
        merge_ref_frame = method in (1, 2)
        if method in (1, 2):
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

    d_size = get_deepex_size(render_speed=render_speed.lower(), enable_resize=enable_resize, ex_model=ex_model)
    smc = SmartResizeColorizer(clip_size=d_size, ex_model=ex_model)
    smr = SmartResizeReference(clip_size=d_size, ex_model=ex_model)

    if method != 0 and not (sc_framedir is None):
        if method in (1, 2):
            clip_ref = vs_ext_reference_clip(clip_ref, sc_framedir=sc_framedir, clip_resize=(ex_model == 2))
        else:
            clip_ref = vs_ext_reference_clip(clip, sc_framedir=sc_framedir, clip_resize=(ex_model == 2))

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
    ref_same_as_video = method == 3  # unico caso in cui è True il flag
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
            case 2:  # DeepRemaster
                clip_colored = vs_deepremaster(clip, clip_ref, clip_sc, render_vivid=render_vivid,
                                               ref_weight=ref_weight, memory_size=max_memory_frames)
            case _:
                HAVC_LogMessage(MessageType.EXCEPTION, "HybridAVC: unknown exemplar model id: " + str(ex_model))

    clip_resized = smc.restore_clip_size(clip_colored)

    # restore original resolution details, 5% faster than ShufflePlanes()
    if not (sc_framedir is None) and method == 0 and only_ref_frames:
        # ref frames are saved if sc_framedir is set
        return vs_sc_recover_clip_luma(clip_orig, clip_resized, scenechange=True, sc_framedir=sc_framedir)
    else:
        return vs_recover_clip_luma(clip_orig, clip_resized)


def HAVC_restore_video(clip: vs.VideoNode = None, clip_ref: vs.VideoNode = None, method: int = 6,
                       render_speed: str = 'medium', ex_model: int = 0, ref_merge: int = 0, ref_weight: float = None,
                       ref_thresh: float = None, ref_freq: int = None, ref_norm: bool = False,
                       max_memory_frames: int = 0, render_vivid: bool = True, encode_mode: int = 0,
                       torch_dir: str = model_dir) -> vs.VideoNode:
    """Colorization Function using DeepRemaster/ColorMNet to restore external video provided externally in clip_ref

    :param clip:                Clip to process. Only RGB24 format is supported
    :param clip_ref:            Clip containing the reference frames (necessary if method=0,1,2,5,6)
    :param render_speed:        Preset to control the render method and speed:
                                Allowed values are:
                                        'Fast'   (colors are more washed out)
                                        'Medium' (colors are a little washed out)
                                        'Slow'   (colors are a little more vivid)
    :param ex_model:            "Exemplar-based" model to use for the color propagation, available models are:
                                    0 : ColorMNet (default)
                                    1 : Deep-Exemplar
                                    2 : Deep-Remaster
    :param method:              Method to use to generate reference frames (RF) for the merge.
                                        0 = HAVC same as video
                                        1 = HAVC + RF same as video
                                        2 = HAVC + RF different from video
                                        5 = HAVC restore same as video
                                        6 = HAVC restore different from video (default)
    :param ref_merge:          Method used by DeepEx to merge the reference frames with the frames propagated by DeepEx.
                                It is applicable only with DeepEx method: 0, 1, 5.
                                The HAVC reference frames must be produced with frequency = 1.
                                Allowed values are:
                                        0 = No RF merge (reference frames can be produced with any frequency)
                                        1 = RF-Merge VeryLow (reference frames are merged with weight=0.3)
                                        2 = RF-Merge Low (reference frames are merged with weight=0.4)
                                        3 = RF-Merge Med (reference frames are merged with weight=0.5)
                                        4 = RF-Merge High (reference frames are merged with weight=0.6)
                                        5 = RF-Merge VeryHigh (reference frames are merged with weight=0.7)
    :param ref_weight:          If (ref_merge > 0), represent the weight used to merge the reference frames.
                                If is not set, is assigned automatically a value depending on ref_merge/method value.
    :param ref_thresh:          Represent the threshold used to create the reference frames. If is not set, is assigned
                                automatically a value of 0.10
    :param ref_freq:            If > 0 will be generated at least a reference frame every "ref_freq" frames.
                                range [0-1500]. If is not set, is assigned automatically a value depending on ref_merge
                                value and method.
    :param ref_norm:            If true the B&W frames are normalized before apply the Scene Detection to generate the
                                reference frames. The normalization will increase the sensitivity to smooth scene changes,
                                range [True, False], default: False
    :param max_memory_frames:   Parameter used by ColorMNet/DeepRemaster models.
                                For ColorMNet specify the max number of encoded frames to keep in memory.
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
                                If = 0 will be filled with the max value (depending on total GPU RAM available).
                                For DeepRemaster represent the number to reference frames to keep in memory.
                                Suggested values are:
                                    min=2, max=50
    :param render_vivid:        Depending on selected ex_model, if enabled (True):
                                    0) ColorMNet: the frames memory is reset at every reference frame update
                                    1) Deep-Exemplar: the saturation will be increased by about 25%.
                                    2) Deep-Remaster: the saturation will be increased by about 15%.
                                range [True, False]
     :param encode_mode:        Parameter used by ColorMNet to define the encode mode strategy.
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
                                     2: remote all-ref   Same as "remote encoding" but all the available reference frames
                                                         will be used for the inference at the beginning of encoding.
    :param torch_dir:           torch hub dir location, default is model directory, if set to None will switch
                                to torch cache dir
    """
    # disable packages warnings
    disable_warnings()

    if method not in (5, 6):
        HAVC_LogMessage(MessageType.EXCEPTION, "HAVC: Video restore is supported only with methods: 5, 6")

    if not isinstance(clip, vs.VideoNode):
        HAVC_LogMessage(MessageType.EXCEPTION, "HAVC_deepex: this is not a clip")

    if not torch.cuda.is_available():
        HAVC_LogMessage(MessageType.EXCEPTION, "HAVC_deepex: CUDA is not available")

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

    if clip_ref.width != clip.width or clip_ref.height != clip.height:
        clip_ref = vs.core.resize.Spline36(clip=clip_ref, width=clip.width, height=clip.height)

    if torch_dir is not None:
        torch.hub.set_dir(torch_dir)

    # static params
    enable_resize = False

    if ref_thresh is None or ref_thresh == 0:
        ref_thresh = DEF_THRESHOLD
    if ref_freq is None or ref_freq == 0:
        if ex_model == 2:
            ref_freq = DEF_MIN_FREQ
        else:
            ref_freq = 0

    refmerge_weight: list[float] = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7]
    if ref_merge == 0 or method in (2, 6):
        clip_ref = SceneDetect(clip_ref, threshold=ref_thresh, frequency=ref_freq, frame_norm=ref_norm)
        ref_weight = 1.0
        clip_sc = None
    else:
        if ref_weight is None or ref_weight == 0:
            ref_weight = refmerge_weight[ref_merge]
        clip_ref = SceneDetect(clip_ref, threshold=0, frequency=1)
        clip_sc = SceneDetect(clip_ref, threshold=ref_thresh, frequency=ref_freq, frame_norm=ref_norm)

    clip = CopySCDetect(clip, clip_ref)

    clip_orig = clip

    d_size = get_deepex_size(render_speed=render_speed.lower(), enable_resize=enable_resize, ex_model=ex_model)
    smc = SmartResizeColorizer(clip_size=d_size, ex_model=ex_model)
    smr = SmartResizeReference(clip_size=d_size, ex_model=ex_model)

    # clip and clip_ref are resized to match the frame size used for inference
    clip = smc.get_resized_clip(clip)
    clip_ref = smr.get_resized_clip(clip_ref)

    ref_same_as_video = False

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
        case 2:  # DeepRemaster
            clip_colored = vs_deepremaster(clip, clip_ref, clip_sc, render_vivid=render_vivid, ref_weight=ref_weight,
                                           memory_size=max_memory_frames, ref_frequency=ref_freq)
        case _:
            HAVC_LogMessage(MessageType.EXCEPTION, "HybridAVC: unknown exemplar model id: " + str(ex_model))

    clip_resized = smc.restore_clip_size(clip_colored)

    # restore original resolution details, 5% faster than ShufflePlanes()
    return vs_recover_clip_luma(clip_orig, clip_resized)


"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
coloring function with additional pre-process and post-process filters 
"""


def HAVC_colorizer(
        clip: vs.VideoNode, method: int = 2, mweight: float = 0.4, deoldify_p: list = (0, 24, 1.0, 0.0),
        ddcolor_p: list = (1, 24, 1.0, 0.0, True), ddtweak: bool = False,
        ddtweak_p: list = (0.0, 1.0, 2.5, True, 0.3, 0.6, 1.5, 0.5, "300:360|0.8,0.1"),
        cmc_tresh: float = 0.2, lmm_p: list = (0.2, 0.8, 1.0), alm_p: list = (0.8, 1.0, 0.15), cmb_sw: bool = False,
        sc_threshold: float = 0.0, sc_tht_offset: int = 1, sc_min_freq: int = 0, sc_tht_ssim: float = 0.0,
        sc_normalize: bool = False, sc_min_int: int = 1, sc_tht_white: float = DEF_THT_WHITE,
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
        HAVC_LogMessage(MessageType.EXCEPTION, "HAVC_colorizer: CUDA is not available")

    if not isinstance(clip, vs.VideoNode):
        HAVC_LogMessage(MessageType.EXCEPTION, "HAVC_colorizer: this is not a clip")

    if sc_threshold < 0:
        HAVC_LogMessage(MessageType.EXCEPTION, "HAVC_colorizer: sc_threshold must be >= 0")

    if sc_min_freq < 0:
        HAVC_LogMessage(MessageType.EXCEPTION, "HAVC_colorizer: sc_min_freq must be >= 0")

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
        HAVC_LogMessage(MessageType.EXCEPTION, "HAVC_colorizer: model files have not been downloaded.")

    if device_index > 7 and device_index != 99:
        HAVC_LogMessage(MessageType.EXCEPTION, "HAVC_colorizer: wrong device_index, choices are: GPU0...GPU7, CPU=99")

    if ddcolor_rf != 0 and ddcolor_rf not in range(10, 65):
        HAVC_LogMessage(MessageType.EXCEPTION, "HAVC_colorizer: ddcolor render_factor must be between: 10-64")

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

    if scenechange:
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
function with HAVC merge methods  
"""


def HAVC_merge(clipa: vs.VideoNode, clipb: vs.VideoNode, clip_luma: vs.VideoNode = None, weight: float = 0.5,
               method: int = 2, cmc_tresh: float = 0.2, lmm_p: list = (0.2, 0.8, 1.0), alm_p: list = (0.8, 1.0, 0.25)
               ) -> vs.VideoNode:
    """Utility function with the implementation of HAVC merge methods

    :param clipa:               first clip to merge, only RGB24 format is supported
    :param clipb:               second clip to merge, only RGB24 format is supported
    :param method:              method used to combine clipa with clipb (default = 2):
                                    0 : clipa only (no merge)
                                    1 : clipb only (no merge)
                                    2 : Simple Merge (default):
                                        the frames are combined using a weighted merge, where the parameter "weight"
                                        represent the weight assigned to the colors provided by the clipb frames.
                                        If weight = 0 will be returned clipa, if = 1 will be returned clipb.
                                    3 : Constrained Chroma Merge:
                                        The frames are combined by assigning a limit to the amount of difference in
                                        chroma values between clipa and clipb this limit is defined by the threshold
                                        parameter "cmc_tresh".
                                        The limit is applied to the image converted to "YUV". For example when
                                        cmc_tresh=0.2, the chroma values "U","V" of clipb frame will be constrained
                                        to have an absolute percentage difference respect to "U","V" provided by clipa
                                        not higher than 20%. The final limited frame will be merged again with the clipa
                                        frame. With this method is suggested a starting weight > 50% (ex. = 60%).
                                    4 : Luma Masked Merge:
                                        the frames are combined using a masked merge, the pixels of clipb with
                                        luma < "luma_mask_limit" will be filled with the pixels of clipa.
                                        If "luma_white_limit" > "luma_mask_limit" the mask will apply a gradient till
                                        "luma_white_limit". If the parameter "weight" > 0 the final masked frame will
                                        be merged again with the clipa frame.
                                    5 : Adaptive Luma Merge:
                                        The frames are combined by decreasing the weight assigned to clipb when the
                                        luma is below a given threshold given by: luma_threshold. The weight is
                                        calculated using the formula:
                                            merge_weight = max(weight * (luma/luma_threshold)^alpha, min_weight).
                                        For example with: luma_threshold = 0.6 and alpha = 1, the weight assigned to
                                        clipb will start to decrease linearly when the luma < 60% till "min_weight".
                                        For alpha=2, begins to decrease quadratically (because luma/luma_threshold < 1).
                                    The methods 3 and 4 are similar to Simple Merge, but before the merge with clipa
                                    the clipb frame is limited in the chroma changes (method 3) or limited based on the
                                    luma (method 4). The method 5 is a Simple Merge where the weight decrease with luma.
    :param weight:              weight given to clipb in all merge methods. If weight = 0 will be returned
                                clipa, if = 1 will be returned clipb. range [0-1] (0.01=1%)
    :param cmc_tresh:           chroma_threshold (%), used by: "Constrained Chroma Merge", range [0-1] (0.01=1%)
    :param lmm_p:               parameters for method: "Luma Masked Merge" (see method=4 for a full explanation)
                                   [0] : luma_mask_limit: luma limit for build the mask used in Luma Masked Merge,
                                         range [0-1] (0.01=1%)
                                   [1] : luma_white_limit: the mask will apply a gradient till luma_white_limit,
                                         range [0-1] (0.01=1%)
                                   [2] : luma_mask_sat: if < 1 the clipb dark pixels will substitute with the
                                         desaturated clipa pixels, range [0-1] (0.01=1%)
    :param alm_p:               parameters for method: "Adaptive Luma Merge" (see method=5 for a full explanation)
                                   [0] : luma_threshold: threshold for the gradient merge, range [0-1] (0.01=1%)
                                   [1] : alpha: exponent parameter used for the weight calculation, range [>0]
                                   [2] : min_weight: min merge weight, range [0-1] (0.01=1%)
    :param clip_luma:           if specified, clip_luma will be used as source of luma component for the merge. It is an
                                optional parameter, and it is suggested to provide the clip with the best luma
                                resolution between clipa and clipb. It is used only with the methods: 3, 4, 5 and can
                                speed up the filter when it uses these methods. 
    """
    # disable packages warnings
    disable_warnings()

    if not isinstance(clipa, vs.VideoNode):
        HAVC_LogMessage(MessageType.EXCEPTION, "HAVC_merge: this is not a clip: clipa")

    if not isinstance(clipb, vs.VideoNode):
        HAVC_LogMessage(MessageType.EXCEPTION, "HAVC_merge: this is not a clip: clipb")

    if method == 0 or weight == 0:
        return clipa

    if method == 1 or weight == 1:
        return clipb

    if method == 2:
        return vs_simple_merge(clipa, clipb, weight)

    merge_weight = weight

    if clipa.format.id != vs.RGB24:
        # clip not in RGB24 format, it will be converted
        if clipa.format.color_family == "YUV":
            clipa = clipa.resize.Bicubic(format=vs.RGB24, matrix_in_s="709", range_s="full",
                                         dither_type="error_diffusion")
        else:
            clipa = clipa.resize.Bicubic(format=vs.RGB24, range_s="full")

    if clipb.format.id != vs.RGB24:
        # clip not in RGB24 format, it will be converted
        if clipb.format.color_family == "YUV":
            clipb = clipb.resize.Bicubic(format=vs.RGB24, matrix_in_s="709", range_s="full",
                                         dither_type="error_diffusion")
        else:
            clipb = clipb.resize.Bicubic(format=vs.RGB24, range_s="full")

    if clip_luma is not None:
        if not isinstance(clip_luma, vs.VideoNode):
            HAVC_LogMessage(MessageType.EXCEPTION, "HAVC_merge: this is not a clip: clip_luma")
        rf = min(max(math.trunc(0.4 * clip_luma.width / 16), 16), 48)
        frame_size = min(rf * 16, clip_luma.width)
        clipa = clipa.resize.Spline64(width=frame_size, height=frame_size)
        clipb = clipb.resize.Spline64(width=frame_size, height=frame_size)

    clip_merged = vs_combine_models(clip_a=clipa, clip_b=clipb, method=method, sat=[1, 1],
                                    hue=[0, 0], clipb_weight=merge_weight, CMC_p=cmc_tresh,
                                    LMM_p=lmm_p, ALM_p=alm_p, invert_clips=False)

    if clip_luma is not None:
        clip_merged = _clip_chroma_resize(clip_luma, clip_merged)

    return clip_merged


"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description: 
------------------------------------------------------------------------------- 
Function to perform colorization with DeepRemaster: : Temporal Source-Reference 
Attention Networks for Comprehensive Video Enhancement.
"""


def HAVC_DeepRemaster(
        clip: vs.VideoNode,
        length: int = 2,
        render_vivid: bool = False,
        ref_dir: str = None,
        ref_minedge: int = 256,
        frame_mindim: int = 320,
        ref_buffer_size: int = 20,
        device_index: int = 0,
        inference_mode: bool = False,
        mode: int = 0
) -> vs.VideoNode:
    """Function to perform colorization with DeepRemaster using direct access to RF or using Vapoursynth clips

     :param clip:            Clip to process. Only RGB24 "full range" format is supported.
     :param length:          Sequence length that the model processes (min. 2, max. 5). Default: 2
     :param render_vivid:    Given that the generated colors by the inference are a little washed out, by enabling
                             this parameter, the saturation will be increased by about 10%. range [True, False]
     :param ref_dir:         Path of the reference frames, must be of the same size of input clip: Default: None
     :param ref_minedge:     min dimension of reference frames used for inference. Default: 256
     :param frame_mindim:    min dimension of input frames used for inference. Default: 320
     :param ref_buffer_size: reference frame buffer size for inference. Default: 20
     :param device_index:    Device ordinal of the GPU (if = -1 CPU mode is enabled). Default: 0
     :param inference_mode:  Enable/Disable torch inference mode. Default: False
     :param mode:            Mode selected to access to the external reference frames.
                             Allowed values are:
                                 0: will use direct access to reference frame folder (fast)
                                 1: will use Vapoursynth clips to access to reference frames (slow)
     """
    if not isinstance(clip, vs.VideoNode):
        raise vs.Error("HAVC_DeepRemaster: this is not a clip")

    if ref_dir is None:
        HAVC_LogMessage(MessageType.EXCEPTION, "HAVC_DeepRemaster: ref_dir is unset")

    if mode == 0:
        return vs_remaster_colorize(clip, length, render_vivid, ref_dir, ref_minedge,
                                    frame_mindim, ref_buffer_size, device_index, inference_mode)

    clip = SceneDetectFromDir(clip, sc_framedir=ref_dir, merge_ref_frame=False, ref_frame_ext=True)

    clip_ref = vs_ext_reference_clip(clip, sc_framedir=ref_dir, clip_resize=True)

    return vs_sc_remaster_colorize(clip, clip_ref, clip_sc=None, length=length, render_vivid=render_vivid,
                                   ref_minedge=ref_minedge, frame_mindim=frame_mindim, ref_buffer_size=ref_buffer_size,
                                   device_index=0, inference_mode=False)


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
                    stab_p: list = (5, 'A', 1, 15, 0.2, 0.8), colormap: str = "none",
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
                                      [5] : tht_scen, threshold for scene change detection (default = 0.8), if=0 is not activated, range [0.01-0.50]
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
                                               tht_scen=stab_tht_scen, algo=stab_algo)

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
Utility function to restore the colors of gray pixels.
"""


def HAVC_recover_clip_color(clip: vs.VideoNode = None, clip_color: vs.VideoNode = None, sat: float = 0.8, tht: int = 30,
                            weight: float = 0.0, alpha: float = 2.0, chroma_resize: bool = True,
                            return_mask: bool = False, binary_mask: bool = False) -> vs.VideoNode:
    """Utility function to restore the colors of gray pixels in the input clip by using the colors provided in the clip:
       clip_color. Useful to repair the clips colored with DeepRemaster

        :param clip:          clip to repair the colors, only RGB24 format is supported
        :param clip_color:    clip with the colors to restore, only RGB24 format is supported
        :param sat:           this parameter allows to change the saturation of colored clip (default = 0.8)
        :param tht:           threshold to identify gray pixels, range[0, 255] (default = 30)
        :param weight:        if > 0, the restored frame will be merged with clip_color's frame. (default = 0.0)
        :param alpha:         parameter used to control the steepness of gradient curve, values above the default value
                              will preserve more pixels, but could introduce some artifacts, range[1, 10] (default = 2)
        :param chroma_resize: if True, the frames will be resized to improve the filter speed (default = True)
        :param return_mask:   if True, will be returned the mask used to identify the gray pixels (white region), could
                              be useful to visualize the gradient mask for debugging, (default = false).
        :param binary_mask:   if True, will be used a binary mask instead of a gradient mask, could be useful to get a
                              clear view on the selected desaturated regions for debugging, (default = false)
    """

    if not isinstance(clip, vs.VideoNode):
        HAVC_LogMessage(MessageType.EXCEPTION, "HAVC_merge: this is not a clip: clip")

    if not isinstance(clip_color, vs.VideoNode):
        HAVC_LogMessage(MessageType.EXCEPTION, "HAVC_merge: this is not a clip: clip_color")

    if clip.format.id != vs.RGB24:
        # clip not in RGB24 format, it will be converted
        if clip.format.color_family == "YUV":
            clip = clip.resize.Bicubic(format=vs.RGB24, matrix_in_s="709", range_s="full",
                                       dither_type="error_diffusion")
        else:
            clip = clip.resize.Bicubic(format=vs.RGB24, range_s="full")

    if clip_color.format.id != vs.RGB24:
        # clip not in RGB24 format, it will be converted
        if clip_color.format.color_family == "YUV":
            clip_color = clip_color.resize.Bicubic(format=vs.RGB24, matrix_in_s="709", range_s="full",
                                                   dither_type="error_diffusion")
        else:
            clip_color = clip_color.resize.Bicubic(format=vs.RGB24, range_s="full")

    alpha = max(min(alpha, DEF_MAX_COLOR_ALPHA), DEF_MIN_COLOR_ALPHA)

    clip_luma = clip
    if chroma_resize and not return_mask:
        rf = min(max(math.trunc(0.4 * clip_luma.width / 16), 16), 48)
        frame_size = min(rf * 16, clip_luma.width)
        clip = clip.resize.Spline64(width=frame_size, height=frame_size)
        clip_color = clip_color.resize.Spline64(width=frame_size, height=frame_size)

    if binary_mask:
        clip_restored = vs_recover_clip_color(clip=clip, clip_color=clip_color, sat=sat, tht=tht, weight=-weight,
                                              tht_scen=1.0, hue_adjust='none', return_mask=return_mask)
    else:
        clip_restored = vs_recover_gradient_color(clip=clip, clip_color=clip_color, sat=sat, tht=tht, weight=weight,
                                                  alpha=alpha, return_mask=return_mask)
    if chroma_resize and not return_mask:
        clip_restored = _clip_chroma_resize(clip_luma, clip_restored)

    return clip_restored


"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description: 
------------------------------------------------------------------------------- 
wrapper to function vSceneDetect() to set the scene-change frames in the clip
"""


def HAVC_SceneDetect(clip: vs.VideoNode, sc_threshold: float = DEF_THRESHOLD, sc_tht_offset: int = 1,
                     sc_tht_ssim: float = 0.0, sc_min_int: int = 1, sc_min_freq: int = 0, sc_normalize: bool = False,
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
                                  sc_framedir: str = "./", sc_sequence: bool = False, sc_normalize: bool = False,
                                  ref_offset: int = 0, sc_tht_white: float = DEF_THT_WHITE,
                                  sc_tht_black: float = DEF_THT_BLACK, ref_ext: str = DEF_EXPORT_FORMAT,
                                  ref_jpg_quality: int = DEF_JPG_QUALITY, ref_override: bool = True,
                                  sc_debug: bool = False) -> vs.VideoNode:
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
    :param sc_sequence:         If True, the reference frames will be exported in sequence, using consecutive numbers.
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
                               ref_jpg_quality=ref_jpg_quality, ref_override=ref_override, sequence=sc_sequence)
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
                                 ref_ext: str = DEF_EXPORT_FORMAT, ref_jpg_quality: int = DEF_JPG_QUALITY,
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
                                   HAVC INTERNAL FUNCTIONS
------------------------------------------------------------------------------------------------------------------------ 
"""

"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description: 
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
Description: 
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
Description: 
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
Description: 
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


"""
------------------------------------------------------------------------------------------------------------------------ 
                                   DDEOLDIFY LEGACY FUNCTIONS (deprecated)
------------------------------------------------------------------------------------------------------------------------ 
"""


def HAVC_ddeoldify(
        clip: vs.VideoNode, method: int = 2, mweight: float = 0.4, deoldify_p: list = (0, 24, 1.0, 0.0),
        ddcolor_p: list = (1, 24, 1.0, 0.0, True), ddtweak: bool = False,
        ddtweak_p: list = (0.0, 1.0, 2.5, True, 0.3, 0.6, 1.5, 0.5, "300:360|0.8,0.1"),
        cmc_tresh: float = 0.2, lmm_p: list = (0.2, 0.8, 1.0), alm_p: list = (0.8, 1.0, 0.15), cmb_sw: bool = False,
        sc_threshold: float = 0.0, sc_tht_offset: int = 1, sc_min_freq: int = 0, sc_tht_ssim: float = 0.0,
        sc_normalize: bool = False, sc_min_int: int = 1, sc_tht_white: float = DEF_THT_WHITE,
        sc_tht_black: float = DEF_THT_BLACK, device_index: int = 0, torch_dir: str = model_dir,
        sc_debug: bool = False) -> vs.VideoNode:
    vs.core.log_message(
        vs.MESSAGE_TYPE_WARNING,
        "Warning: HAVC_ddeoldify is deprecated and may be removed in the future, please use 'HAVC_colorizer' instead.")

    return HAVC_colorizer(clip, method, mweight, deoldify_p, ddcolor_p, ddtweak, ddtweak_p, cmc_tresh, lmm_p, alm_p,
                          cmb_sw, sc_threshold, sc_tht_offset, sc_min_freq, sc_tht_ssim, sc_normalize, sc_min_int,
                          sc_tht_white, sc_tht_black, device_index, torch_dir, sc_debug)


def ddeoldify_main(clip: vs.VideoNode, Preset: str = 'Fast', VideoTune: str = 'Stable', ColorFix: str = 'Violet/Red',
                   ColorTune: str = 'Light', ColorMap: str = 'None', degrain_strength: int = 0,
                   enable_fp16: bool = True) -> vs.VideoNode:
    vs.core.log_message(
        vs.MESSAGE_TYPE_WARNING,
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
    vs.core.log_message(
        vs.MESSAGE_TYPE_WARNING,
        "Warning: ddeoldify is deprecated and may be removed in the future, please use 'HAVC_ddeoldify' instead.")

    return HAVC_colorizer(clip, method, mweight, deoldify_p, ddcolor_p, ddtweak, ddtweak_p, cmc_tresh, lmm_p, alm_p,
                          cmb_sw,
                          sc_threshold=0, sc_min_freq=0, device_index=device_index, torch_dir=torch_dir)


def ddeoldify_stabilizer(clip: vs.VideoNode, dark: bool = False, dark_p: list = (0.2, 0.8), smooth: bool = False,
                         smooth_p: list = (0.3, 0.7, 0.9, 0.0, "none"),
                         stab: bool = False, stab_p: list = (5, 'A', 1, 15, 0.2, 0.80), colormap: str = "none",
                         render_factor: int = 24) -> vs.VideoNode:
    vs.core.log_message(
        vs.MESSAGE_TYPE_WARNING,
        "Warning: ddeoldify_stabilizer is deprecated and may be removed in the future, please use 'HAVC_stabilizer'.")

    return HAVC_stabilizer(clip, dark, dark_p, smooth, smooth_p, stab, stab_p, colormap, render_factor)
