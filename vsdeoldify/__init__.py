"""
------------------------------------------------------------------------------- 
Author: Dan64
Date: 2024-02-29
version: 
LastEditors: Dan64
LastEditTime: 2024-06-10
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
main Vapoursynth wrapper to pytorch-based coloring filter HybridAVC (HAVC).
The filter includes some portions of code from the following coloring projects:
DeOldify: https://github.com/jantic/DeOldify
DDColor: https://github.com/HolyWu/vs-ddcolor
Deep-Exemplar: https://github.com/zhangmozhe/Deep-Exemplar-based-Video-Colorization
"""
from __future__ import annotations
from functools import partial

import os
import pathlib

os.environ["CUDA_MODULE_LOADING"] = "LAZY"
os.environ["NUMEXPR_MAX_THREADS"] = "8"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import math
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transform_lib
from PIL import Image

from vsdeoldify.deoldify import device
from vsdeoldify.deoldify.device_id import DeviceId

from vsdeoldify.vsslib.vsfilters import *
from vsdeoldify.vsslib.mcomb import *
from vsdeoldify.vsslib.vsmodels import *
from vsdeoldify.vsslib.vsresize import SmartResizeColorizer, SmartResizeReference
from vsdeoldify.vsslib.vsutils import SceneDetect

from vsdeoldify.deepex import deepex_colorizer, get_deepex_size, ModelColorizer

__version__ = "4.0.0"

import warnings

warnings.filterwarnings("ignore", category=UserWarning, message=".*?Your .*? set is empty.*?")
warnings.filterwarnings("ignore", category=UserWarning,
                        message="The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.")
warnings.filterwarnings("ignore", category=FutureWarning, message="Arguments other than a weight enum or `None`.*?")
warnings.filterwarnings("ignore", category=UserWarning, message="Arguments other than a weight enum or `None`.*?")
warnings.filterwarnings("ignore", category=UserWarning, message="torch.nn.utils.weight_norm is deprecated.*?")
warnings.filterwarnings("ignore", category=UserWarning, message="Conversion from CIE-LAB,*?")
warnings.filterwarnings("ignore", category=UserWarning, message=".*?Torch was not compiled with flash attention.*?")

package_dir = os.path.dirname(os.path.realpath(__file__))
model_dir = os.path.join(package_dir, "models")

#configuring torch
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


def HAVC_main(clip: vs.VideoNode, Preset: str = 'Fast', VideoTune: str = 'Stable', ColorFix: str = 'Violet/Red',
              ColorTune: str = 'Light', ColorMap: str = 'None', EnableDeepEx: bool = False, DeepExMethod: int = 0,
              DeepExPreset: str = 'Fast', DeepExRefMerge: int = 0, ScFrameDir: str = None, ScThreshold: float = 0.1, ScMinFreq: int = 0,
              enable_fp16: bool = True) -> vs.VideoNode:
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
    :param VideoTune:           Preset to control the output video color stability
                                Allowed values are:
                                    'VeryStable',
                                    'MoreStable'
                                    'Stable',
                                    'Balanced',
                                    'Vivid',
                                    ,MoreVivid'
                                    'VeryVivid',
    :param ColorFix:            This parameter allows to reduce color noise on determinated chroma ranges.
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
    :param EnableDeepEx:        Enable coloring using "Deep-Exemplar-based Video Colorization"
    :param DeepExMethod:        Method to use to generate reference frames.
                                        0 = HAVC (default)
                                        1 = HAVC + RF same as video
                                        2 = HAVC + RF different from video
                                        3 = external RF same as video
                                        4 = external RF different from video
    :param DeepExPreset:        Preset to control the render method and speed:
                                Allowed values are:
                                        'Fast'   (colors are more washed out)
                                        'Medium' (colors are a little washed out)
                                        'Slow'   (colors are a little more vivid)
    :param DeepExRefMerge:      Method used by DeepEx to merge the reference frames with the frames propagated by DeepEx.
                                It is applicable only with DeepEx method: 0, 1, 2.
                                Allowed values are:
                                        0 = No RF merge (reference frames can be produced with any frequency)
                                        1 = RF-Merge Low (reference frames are merged with low weight)
                                        2 = RF-Merge Med. (reference frames are merged with medium weight)
                                        3 = RF-Merge High (reference frames are merged with high weight)
    :param ScFrameDir:          if set, define the directory where are stored the reference frames that will be used
                                by "Deep-Exemplar-based Video Colorization".
    :param ScThreshold:         Scene change threshold used to generate the reference frames to be used by
                                "Deep-Exemplar-based Video Colorization". It is a percentage of the luma change between
                                the previous and the current frame. range [0-1], default 0.10. If =0 are not generate
                                reference frames.
    :param ScMinFreq:           if > 0 will be generated at least a reference frame every "ScMinFreq" frames.
                                range [0-1500], default: 0.
    :param enable_fp16:         Enable/disable FP16 in ddcolor inference, range [True, False]
    """
    # Select presets / tuning
    Preset = Preset.lower()
    presets = ['placebo', 'veryslow', 'slower', 'slow', 'medium', 'fast', 'faster', 'veryfast']
    preset0_rf = [34, 32, 30, 28, 26, 24, 20, 16]
    preset1_rf = [48, 44, 36, 32, 28, 24, 20, 16]

    try:
        pr_id = presets.index(Preset)
    except ValueError:
        raise vs.Error("HAVC_main: Preset choice is invalid for '" + pr_id + "'")

    deoldify_rf = preset0_rf[pr_id]
    ddcolor_rf = preset1_rf[pr_id]

    # vs.core.log_message(2, "Preset index: " + str(pr_id) )

    # Select VideoTune
    VideoTune = VideoTune.lower()
    video_tune = ['verystable', 'morestable', 'stable', 'balanced', 'vivid', 'morevivid', 'veryvivid']
    ddcolor_weight = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    try:
        w_id = video_tune.index(VideoTune)
    except ValueError:
        raise vs.Error("HAVC_main: VideoTune choice is invalid for '" + VideoTune + "'")

        # Select ColorTune for ColorFix
    ColorTune = ColorTune.lower()
    color_tune = ['light', 'medium', 'strong']
    hue_tune = ["0.8,0.1", "0.5,0.1", "0.2,0.1"]
    hue_tune2 = ["0.9,0", "0.7,0", "0.5,0"]

    try:
        tn_id = color_tune.index(ColorTune)
    except ValueError:
        raise vs.Error("HAVC_main: ColorTune choice is invalid for '" + ColorTune + "'")

    # Select ColorFix for ddcolor/stabilizer
    ColorFix = ColorFix.lower()
    color_fix = ['none', 'magenta', 'magenta/violet', 'violet', 'violet/red', 'blue/magenta', 'yellow', 'yellow/orange',
                 'yellow/green']
    hue_fix = ["none", "270:300", "270:330", "300:330", "300:360", "220:280", "60:90", "30:90", "60:120"]

    try:
        co_id = color_fix.index(ColorFix)
    except ValueError:
        raise vs.Error("HAVC_main: ColorFix choice is invalid for '" + ColorFix + "'")

    if co_id == 0:
        hue_range = "none"
        hue_range2 = "none"
    else:
        hue_range = hue_fix[co_id] + "|" + hue_tune[tn_id]
        hue_range2 = hue_fix[co_id] + "|" + hue_tune2[tn_id]

    # Select Color Mapping
    ColorMap = ColorMap.lower()
    colormap = ['none', 'blue->brown', 'blue->red', 'blue->green', 'green->brown', 'green->red', 'green->blue',
                'red->brown', 'red->blue', 'yellow->rose']
    hue_map = ["none", "180:280|+140,0.4", "180:280|+100,0.4", "180:280|+220,0.4", "80:180|+260,0.4", "80:180|+220,0.4",
               "80:180|+140,0.4", "300:360,0:20|+40,0.6", "300:360,0:20|+260,0.6", "30:90|+300,0.8"]

    try:
        cl_id = colormap.index(ColorMap)
    except ValueError:
        raise vs.Error("HAVC_main: ColorMap choice is invalid for '" + ColorMap + "'")

    chroma_adjust = hue_map[cl_id]

    if EnableDeepEx and DeepExMethod in (0, 1, 2):

        if DeepExRefMerge > 0:
            clip_ref = HAVC_ddeoldify(clip, method=1, ddcolor_p=[1, ddcolor_rf, 1.0, 0.0, enable_fp16],
                                  ddtweak=True, ddtweak_p=[0.0, 1.0, 2.5, True, 0.3, 0.6, 1.5, 0.5, hue_range],
                                  sc_threshold=ScThreshold, sc_min_freq=1)
        else:
            clip_ref = HAVC_ddeoldify(clip, method=2, mweight=ddcolor_weight[w_id],
                                      deoldify_p=[0, deoldify_rf, 1.0, 0.0],
                                      ddcolor_p=[1, ddcolor_rf, 1.0, 0.0, enable_fp16],
                                      ddtweak=True, ddtweak_p=[0.0, 1.0, 2.5, True, 0.3, 0.6, 1.5, 0.5, hue_range],
                                      sc_threshold=ScThreshold, sc_min_freq=ScMinFreq)

        clip_colored = HAVC_deepex(clip=clip, clip_ref=clip_ref, method=DeepExMethod, render_speed=DeepExPreset,
                                   render_vivid=True, ref_merge=DeepExRefMerge, sc_framedir=ScFrameDir,
                                   only_ref_frames=False, dark=True, dark_p=[0.2, 0.8],
                                   smooth=True, smooth_p=[0.3, 0.7, 0.9, 0.0, "none"], colormap=chroma_adjust)

        if ScMinFreq in range(1, 20):
            clip_colored = HAVC_stabilizer(clip_colored, stab_p=[5, 'A', 1, 15, 0.2, 0.15])
        else:
            clip_colored = HAVC_stabilizer(clip_colored, stab_p=[3, 'A', 1, 0, 0, 0])

    elif EnableDeepEx and DeepExMethod in (3, 4):

        clip_colored = HAVC_deepex(clip=clip, clip_ref=None, method=DeepExMethod, render_speed=DeepExPreset,
                                   render_vivid=True, sc_framedir=ScFrameDir, only_ref_frames=False, dark=True,
                                   dark_p=[0.2, 0.8], smooth=True, smooth_p=[0.3, 0.7, 0.9, 0.0, "none"],
                                   colormap=chroma_adjust)

    else:
        clip_colored = HAVC_ddeoldify(clip, method=2, mweight=ddcolor_weight[w_id],
                                      deoldify_p=[0, deoldify_rf, 1.0, 0.0],
                                      ddcolor_p=[1, ddcolor_rf, 1.0, 0.0, enable_fp16],
                                      ddtweak=True, ddtweak_p=[0.0, 1.0, 2.5, True, 0.3, 0.6, 1.5, 0.5, hue_range])

        if pr_id > 5 and cl_id > 0:
            clip_colored = HAVC_stabilizer(clip_colored, colormap=chroma_adjust)
        elif pr_id > 3:
            clip_colored = HAVC_stabilizer(clip_colored, dark=True, dark_p=[0.2, 0.8],
                                           smooth=True, smooth_p=[0.3, 0.7, 0.9, 0.0, chroma_adjust],
                                           stab=True, stab_p=[5, 'A', 1, 15, 0.2, 0.15])
        else:
            clip_colored = HAVC_stabilizer(clip_colored, dark=True, dark_p=[0.2, 0.8],
                                           smooth=True, smooth_p=[0.3, 0.7, 0.9, 0.0, chroma_adjust],
                                           stab=True, stab_p=[5, 'A', 1, 15, 0.2, 0.15, hue_range2])
    return clip_colored


"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
Deep-Exemplar coloring function with additional post-process filters 
"""


def HAVC_deepex(clip: vs.VideoNode = None, clip_ref: vs.VideoNode = None, method: int = 0, render_speed: str = 'medium',
                render_vivid: bool = True, ref_merge: int = 0, sc_framedir: str = None,
                only_ref_frames: bool = False, dark: bool = False, dark_p: list = [0.2, 0.8], smooth: bool = False,
                smooth_p: list = [0.3, 0.7, 0.9, 0.0, "none"], colormap: str = "none",
                ref_weight: float = None, ref_thresh: float = None) -> vs.VideoNode:
    """Towards Video-Realistic Colorization via Deep Exemplar-based framework

    :param clip:                Clip to process. Only RGB24 format is supported
    :param clip_ref:            Clip containing the reference frames (necessary if method=0,1,2)
    :param method:              Method to use to generate reference frames (RF).
                                        0 = HAVC (default)
                                        1 = HAVC + RF same as video
                                        2 = HAVC + RF different from video
                                        3 = external RF same as video
                                        4 = external RF different from video
    :param render_speed:        Preset to control the render method and speed:
                                Allowed values are:
                                        'Fast'   (colors are more washed out)
                                        'Medium' (colors are a little washed out)
                                        'Slow'   (colors are a little more vivid)
    :param render_vivid:        Given that the generated colors by the inference are a little washed out, by enabling
                                this parameter, the saturation will be increased by about 25%. range [True, False]
    :param ref_merge:           Method used by DeepEx to merge the reference frames with the frames propagated by DeepEx.
                                It is applicable only with DeepEx method: 0, 1, 2.
                                The HAVC reference frames must be produced with frequency = 1.
                                Allowed values are:
                                        0 = No RF merge (reference frames can be produced with any frequency)
                                        1 = RF-Merge Low (reference frames are merged with low weight)
                                        2 = RF-Merge Med. (reference frames are merged with medium weight)
                                        3 = RF-Merge High (reference frames are merged with high weight)
    :param sc_framedir:         If set, define the directory where are stored the reference frames. If only_ref_frames=True,
                                and method=0 this directory will be written with the reference frames used by the filter.
                                if method!=0 the directory will be read to create the reference frames that will be used
                                by "Deep-Exemplar-based Video Colorization". The reference frame name must be in the
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
    :param ref_weight:          If enable_refmerge = True, represent the weight used to merge the reference frames.
                                If is not set, is assigned automatically a value of 0.5
    :param ref_thresh:          If enable_refmerge = True, represent the threshold used to create the reference frames.
                                If is not set, is assigned automatically a value of 0.1
    """
    if not isinstance(clip, vs.VideoNode):
        raise vs.Error("HAVC_deepex: this is not a clip")

    if not torch.cuda.is_available():
        raise vs.Error("HAVC_deepex: CUDA is not available")

    if only_ref_frames and (sc_framedir is None):
        raise vs.Error("HAVC_deepex: only_ref_frames is enabled but sc_framedir is unset")

    if method != 0 and (sc_framedir is None):
        raise vs.Error("HAVC_deepex: method != 0 but sc_framedir is unset")

    if method in (3, 4) and not (clip_ref is None):
        raise vs.Error("HAVC_deepex: method in (3, 4) but clip_ref is set")

    if method in (0, 1, 2) and (clip_ref is None):
        raise vs.Error("HAVC_deepex: method in (0, 1, 2) but clip_ref is unset")

    # creates the directory "sc_framedir" and does not raise an exception if the directory already exists
    if not (sc_framedir is None):
        pathlib.Path(sc_framedir).mkdir(parents=True, exist_ok=True)

    if clip.format.id != vs.RGB24:
        # clip not in RGB24 format, it will be converted
        if clip.format.color_family == "YUV":
            clip = clip.resize.Bicubic(format=vs.RGB24, matrix_in_s="709", range_s="full",
                                       dither_type="error_diffusion")
        else:
            clip = clip.resize.Bicubic(format=vs.RGB24, range_s="full")

    if method not in range(5):
        raise vs.Error("HAVC_deepex: method must be in range [0-4]")

    if ref_merge not in range(4):
        raise vs.Error("HAVC_deepex: method must be in range [0-3]")

    if ref_merge > 0 and method > 2:
        raise vs.Error("HAVC_deepex: method must be in range [0-2] to be used with ref_merge > 0")

    if method in (0, 1, 2):
        sc_threshold, sc_frequency = get_sc_props(clip_ref)
        if sc_threshold == 0 and sc_frequency == 0:
            raise vs.Error("HAVC_deepex: method in (0, 1, 2) but sc_threshold and sc_frequency are not set")

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

    enable_refmerge = (ref_merge > 0)
    refmerge_weight = [0.0, 0.4, 0.5, 0.6]
    if enable_refmerge:
        if ref_weight is None:
            ref_weight = refmerge_weight[ref_merge]
        if ref_thresh is None:
            ref_thresh = 0.1
        clip_sc = SceneDetect(clip, threshold=ref_thresh)
        if method in (1, 2):
            clip_sc = SceneDetectFromDir(clip_sc, sc_framedir=sc_framedir, merge_ref_frame=True,
                                         ref_frame_ext=(method == 2))
    else:
        ref_weight = 1.0
        clip_sc = None

    if method != 0:
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

    d_size = get_deepex_size(render_speed=render_speed.lower(), enable_resize=enable_resize)
    smc = SmartResizeColorizer(d_size)
    smr = SmartResizeReference(d_size)

    if method != 0:
        if method in (1, 2):
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

    frame_propagate = (method != 2)

    if only_ref_frames:
        clip_colored = clip_ref
    else:
        clip_colored = vs_deepex(clip, clip_ref, clip_sc, image_size=d_size, enable_resize=enable_resize,
                                 wls_filter_on=True, frame_propagate=frame_propagate, render_vivid=render_vivid,
                                 ref_weight=ref_weight)

    clip_resized = smc.restore_clip_size(clip_colored)

    # restore original resolution details, 5% faster than ShufflePlanes()
    if not (sc_framedir is None) and method == 0:
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
        clip: vs.VideoNode, method: int = 2, mweight: float = 0.4, deoldify_p: list = [0, 24, 1.0, 0.0],
        ddcolor_p: list = [1, 24, 1.0, 0.0, True], ddtweak: bool = False,
        ddtweak_p: list = [0.0, 1.0, 2.5, True, 0.3, 0.6, 1.5, 0.5, "300:360|0.8,0.1"],
        cmc_tresh: float = 0.2, lmm_p: list = [0.2, 0.8, 1.0], alm_p: list = [0.8, 1.0, 0.15], cmb_sw: bool = False,
        sc_threshold: float = 0.0, sc_min_freq: int = 0, device_index: int = 0,
        torch_dir: str = model_dir) -> vs.VideoNode:
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
                                   [1] render factor for the model, if=0 will be auto selected
                                       (default = 24), range: [0, 10-64]
                                   [2] saturation parameter to apply to deoldify color model (default = 1)
                                   [3] hue parameter to apply to deoldify color model (default = 0)
                                   [4] enable/disable FP16 in ddcolor inference
    :param ddtweak:             enabled/disable tweak parameters for ddcolor(), range [True,False]
    :param ddtweak_p:           tweak parameters for ddcolor():
                                   [0] : ddcolor tweak's bright (default = 0)
                                   [1] : ddcolor tweak's constrast (default = 1), if < 1 ddcolor provides de-saturated frames
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
    :param cmc_tresh:           chroma_threshold (%), used by: Constrained "Chroma Merge range" [0-1] (0.01=1%)
    :param lmm_p:               parameters for method: "Luma Masked Merge" (see method=4 for a full explanation)
                                   [0] : luma_mask_limit: luma limit for build the mask used in Luma Masked Merge, range [0-1] (0.01=1%)
                                   [1] : luma_white_limit: the mask will appliey a gradient till luma_white_limit, range [0-1] (0.01=1%)
                                   [2] : luma_mask_sat: if < 1 the ddcolor dark pixels will substitute with the desaturated deoldify pixels, range [0-1] (0.01=1%)
    :param alm_p:               parameters for method: "Adaptive Luma Merge" (see method=5 for a full explanation)
                                   [0] : luma_threshold: threshold for the gradient merge, range [0-1] (0.01=1%)
                                   [1] : alpha: exponent parameter used for the weight calculation, range [>0]
                                   [2] : min_weight: min merge weight, range [0-1] (0.01=1%)
    :param cmb_sw:              if true switch the clip order in all the combining methods, range [True,False]
    :param sc_threshold:        Scene change threshold used to generate the reference frames to be used by
                                "Deep-Exemplar-based Video Colorization". It is a percentage of the luma change between
                                the previous and the current frame. range [0-1], default 0.0. If =0 are not generate
                                reference frames and will colorized all the frames.
    :param sc_min_freq:         if > 0 will be generate at least a reference frame every "sc_min_freq" frames.
                                range [0-1500], default: 0.
    :param device_index:        device ordinal of the GPU, choices: GPU0...GPU7, CPU=99 (default = 0)
    :param torch_dir:           torch hub dir location, default is model directory, if set to None will switch to torch cache dir.
    """

    if (not torch.cuda.is_available() and device_index != 99):
        raise vs.Error("HAVC_ddeoldify: CUDA is not available")

    if not isinstance(clip, vs.VideoNode):
        raise vs.Error("HAVC_ddeoldify: this is not a clip")

    if sc_threshold < 0:
        raise vs.Error("HAVC_ddeoldify: sc_threshold must be >= 0")

    if sc_min_freq < 0:
        raise vs.Error("HAVC_ddeoldify: sc_min_freq must be >= 0")

    merge_weight = mweight

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
        raise vs.Error("HAVC_ddeoldify: model files have not been downloaded.")

    if device_index > 7 and device_index != 99:
        raise vs.Error("HAVC_ddeoldify: wrong device_index, choices are: GPU0...GPU7, CPU=99")

    if ddcolor_rf != 0 and ddcolor_rf not in range(10, 65):
        raise vs.Error("HAVC_ddeoldify: ddcolor render_factor must be between: 10-64")

    if clip.format.id != vs.RGB24:
        # clip not in RGB24 format, it will be converted
        if (clip.format.color_family == "YUV"):
            clip = clip.resize.Bicubic(format=vs.RGB24, matrix_in_s="709", range_s="full",
                                       dither_type="error_diffusion")
        else:
            clip = clip.resize.Bicubic(format=vs.RGB24, range_s="full")

            # choices: GPU0...GPU7, CPU=99
    device.set(device=DeviceId(device_index))

    if torch_dir != None:
        torch.hub.set_dir(torch_dir)

    if ddcolor_rf == 0:
        ddcolor_rf = min(max(math.trunc(0.4 * clip.width / 16), 16), 48)

    scenechange = not (sc_threshold == 0 and sc_min_freq == 0)

    clip = SceneDetect(clip, threshold=sc_threshold, frequency=sc_min_freq)

    frame_size = min(max(ddcolor_rf, deoldify_rf) * 16, clip.width)  # frame size calculation for inference()
    clip_orig = clip;
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

    return _clip_chroma_resize(clip_orig, clip_colored)


"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
Video color stabilization filter.
"""


def HAVC_stabilizer(clip: vs.VideoNode, dark: bool = False, dark_p: list = [0.2, 0.8], smooth: bool = False,
                    smooth_p: list = [0.3, 0.7, 0.9, 0.0, "none"], stab: bool = False,
                    stab_p: list = [5, 'A', 1, 15, 0.2, 0.15], colormap: str = "none",
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
                                      [2] : sat: saturation applied to the restored gray prixels [0,1]
                                      [3] : tht, threshold to detect gray pixels, range [0,255], if=0 is not applied the restore,
                                            its value depends on merge method used, suggested values are:
                                                method 0: tht = 5
                                                method 1: tht = 60 (ddcolor provides very saturared frames)
                                                method 2: tht = 15
                                                method 3: tht = 20
                                                method 4: tht = 5
                                                method 5: tht = 10
                                      [4] : weight, weight to blend the restored imaage (default=0.2), range [0-1], if=0 is not applied the blending
                                      [5] : tht_scen, threshold for scene change detection (default = 0.15), if=0 is not activated, range [0.01-0.50]
                                      [6] : "chroma adjustment" parameter (optional), if="none" is disabled (see the README)
        :param colormap:             direct hue/color mapping, without luma filtering, using the "chroma adjustment" parameter, if="none" is disabled
        :param render_factor:       render_factor to apply to the filters, the frame size will be reduced to speed-up the filters,
                                    but the final resolution will be the one of the original clip. If = 0 will be auto selected.
                                    This approach takes advantage of the fact that human eyes are much less sensitive to
                                    imperfections in chrominance compared to luminance. This means that it is possible to speed-up
                                    the chroma filters and get a great high-resolution result in the end, range: [0, 10-64]
    """

    if not isinstance(clip, vs.VideoNode):
        raise vs.Error("HAVC_stabilizer: this is not a clip")

    if clip.format.id != vs.RGB24:
        # clip not in RGB24 format, it will be converted
        if (clip.format.color_family == "YUV"):
            clip = clip.resize.Bicubic(format=vs.RGB24, matrix_in_s="709", range_s="full",
                                       dither_type="error_diffusion")
        else:
            clip = clip.resize.Bicubic(format=vs.RGB24, range_s="full")

            # enable chroma_resize
    chroma_resize_enabled = True

    if render_factor != 0 and render_factor not in range(16, 65):
        raise vs.Error("HAVC_stabilizer: render_factor must be between: 16-64")

    if render_factor == 0:
        render_factor = min(max(math.trunc(0.4 * clip.width / 16), 16), 64)

    if chroma_resize_enabled:
        frame_size = min(render_factor * 16, clip.width)  # frame size calculation for filters
        clip_orig = clip;
        clip = clip.resize.Spline64(width=frame_size, height=frame_size)

    # unpack dark
    dark_enabled = dark
    dark_threshold = dark_p[0]
    dark_amount = dark_p[1]
    if (len(dark_p) > 2):
        dark_hue_adjust = dark_p[2]
    else:
        dark_hue_adjust = 'none'

    # unpack chroma_smoothing
    chroma_smoothing_enabled = smooth
    black_threshold = smooth_p[0]
    white_threshold = smooth_p[1]
    dark_sat = smooth_p[2]
    dark_bright = -smooth_p[3]  # change the sign to reduce the bright
    if (len(smooth_p) > 4):
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
------------------------------------------------------------------------------------------------------------------------ 
                                   DDEOLDIFY LEGACY FUNCTIONS (deprecated)
------------------------------------------------------------------------------------------------------------------------ 
"""


def ddeoldify_main(clip: vs.VideoNode, Preset: str = 'Fast', VideoTune: str = 'Stable', ColorFix: str = 'Violet/Red',
                   ColorTune: str = 'Light', ColorMap: str = 'None', degrain_strength: int = 0,
                   enable_fp16: bool = True) -> vs.VideoNode:
    vs.core.log_message(2,
                        "Warning: ddeoldify_main is deprecated and may be removed in the future, please use 'HAVC_main' instead.")

    return HAVC_main(clip=clip, Preset=Preset, VideoTune=VideoTune, ColorFix=ColorFix, ColorTune=ColorTune,
                     ColorMap=ColorMap, enable_fp16=enable_fp16)


def ddeoldify(clip: vs.VideoNode, method: int = 2, mweight: float = 0.4, deoldify_p: list = [0, 24, 1.0, 0.0],
              ddcolor_p: list = [1, 24, 1.0, 0.0, True],
              dotweak: bool = False, dotweak_p: list = [0.0, 1.0, 1.0, False, 0.2, 0.5, 1.5, 0.5],
              ddtweak: bool = False, ddtweak_p: list = [0.0, 1.0, 2.5, True, 0.3, 0.6, 1.5, 0.5, "300:360|0.8,0.1"],
              degrain_strength: int = 0, cmc_tresh: float = 0.2, lmm_p: list = [0.2, 0.8, 1.0],
              alm_p: list = [0.8, 1.0, 0.15], cmb_sw: bool = False, device_index: int = 0,
              torch_dir: str = model_dir) -> vs.VideoNode:
    vs.core.log_message(2,
                        "Warning: ddeoldify is deprecated and may be removed in the future, please use 'HAVC_ddeoldify' instead.")

    return HAVC_ddeoldify(clip, method, mweight, deoldify_p, ddcolor_p, ddtweak, ddtweak_p, cmc_tresh, lmm_p, alm_p,
                          cmb_sw,
                          sc_threshold=0, sc_min_freq=0, device_index=device_index, torch_dir=torch_dir)


def ddeoldify_stabilizer(clip: vs.VideoNode, dark: bool = False, dark_p: list = [0.2, 0.8], smooth: bool = False,
                         smooth_p: list = [0.3, 0.7, 0.9, 0.0, "none"],
                         stab: bool = False, stab_p: list = [5, 'A', 1, 15, 0.2, 0.15], colormap: str = "none",
                         render_factor: int = 24) -> vs.VideoNode:
    vs.core.log_message(2,
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


def _extract_reference_frames(clip: vs.VideoNode, sc_threshold: float = 0.0, sc_min_freq: int = 0,
                              sc_framedir: str = "./") -> vs.VideoNode:

    pathlib.Path(sc_framedir).mkdir(parents=True, exist_ok=True)
    clip = SceneDetect(clip, threshold=sc_threshold, frequency=sc_min_freq)
    clip = vs_sc_export_frames(clip, sc_framedir)
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
