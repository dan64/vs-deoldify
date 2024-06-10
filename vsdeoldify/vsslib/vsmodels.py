"""
------------------------------------------------------------------------------- 
Author: Dan64
Date: 2024-04-08
version: 
LastEditors: Dan64
LastEditTime: 2024-05-26
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
module containing the main functions to colorize the frames with deoldify() and ddcolor().
"""
import vapoursynth as vs
import math
import numpy as np
import cv2
import os
from PIL import Image
from functools import partial

from ..deoldify.visualize import *
from .vsutils import *
from .vsfilters import *
from ..deepex import deepex_colorizer, ModelColorizer


def vs_deepex(clip: vs.VideoNode, clip_ref: vs.VideoNode, clip_sc: vs.VideoNode, image_size: list = [432, 768], enable_resize: bool = False,
              wls_filter_on: bool = True, frame_propagate: bool = True,
              render_vivid: bool = True, ref_weight: float = 1.0) -> vs.VideoNode:

    colorizer = deepex_colorizer(image_size=image_size, enable_resize=enable_resize)

    def deepex_clip_color_merge(n, f, colorizer: ModelColorizer = None, wls_on: bool = True,
                          propagate: bool = True, render_vivid: bool = True, weight: float = 1.0) -> vs.VideoFrame:

        is_scenechange = f[2].props['_SceneChangePrev'] == 1
        img_orig = frame_to_image(f[0])
        img_ref = frame_to_image(f[1])

        if n == 0:
            # vs.core.log_message(2, "Reference Frame: " + str(n))
            colorizer.set_ref_frame(img_ref)
        elif is_scenechange:
            # vs.core.log_message(2, "Reference Frame: " + str(n))
            colorizer.set_ref_frame(img_ref, frame_propagate=True)

        img_color = colorizer.colorize_frame(img_orig, wls_filter_on=wls_on, render_vivid=render_vivid)

        if not is_scenechange:
            img_color_m = image_weighted_merge(img_color, img_ref, weight)
        else:
            img_color_m = img_color

        return image_to_frame(img_color_m, f[0].copy())

    def deepex_clip_color(n, f, colorizer: ModelColorizer = None, wls_on: bool = True,
                          propagate: bool = True, render_vivid: bool = True) -> vs.VideoFrame:

        is_scenechange = f[1].props['_SceneChangePrev'] == 1
        is_scenechange_ext = f[1].props['_SceneChangeNext'] == 1
        img_orig = frame_to_image(f[0])

        if n == 0:
            img_ref = frame_to_image(f[1])
            # vs.core.log_message(2, "Reference Frame: " + str(n))
            colorizer.set_ref_frame(img_ref)
        elif is_scenechange:
            img_ref = frame_to_image(f[1])
            # vs.core.log_message(2, "Reference Frame: " + str(n))
            frame_propagate = not is_scenechange_ext
            colorizer.set_ref_frame(img_ref, frame_propagate)

        img_color = colorizer.colorize_frame(img_orig, wls_filter_on=wls_on, render_vivid=render_vivid)

        return image_to_frame(img_color, f[0].copy())

    if ref_weight < 1 and not (clip_sc is None):
        clip_colored = clip.std.ModifyFrame(clips=[clip, clip_ref, clip_sc], selector=partial(deepex_clip_color_merge, colorizer=colorizer,
                                            wls_on=wls_filter_on, propagate=frame_propagate, render_vivid=render_vivid, weight=ref_weight))
    else:
        clip_colored = clip.std.ModifyFrame(clips=[clip, clip_ref], selector=partial(deepex_clip_color, colorizer=colorizer,
                                            wls_on=wls_filter_on, propagate=frame_propagate, render_vivid=render_vivid))
    return clip_colored


"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
wrapper to deoldify. 
"""


def vs_deoldify(clip: vs.VideoNode, method: int = 2, model: int = 0, render_factor: int = 24, scenechange: bool = True,
                package_dir: str = "") -> vs.VideoNode:
    return vs_sc_deoldify(clip, method, model, render_factor, scenechange=False, package_dir=package_dir)


def vs_sc_deoldify(clip: vs.VideoNode, method: int = 2, model: int = 0, render_factor: int = 24,
                   scenechange: bool = True, package_dir: str = "") -> vs.VideoNode:
    if method == 1:
        return None

    m_cfg = ModelImageInitializer(package_dir=package_dir)

    match model:
        case 0:
            colorizer = m_cfg.get_image_colorizer(artistic=False, isvideo=True)
        case 1:
            colorizer = m_cfg.get_image_colorizer(artistic=False, isvideo=False)
        case 2:
            colorizer = m_cfg.get_image_colorizer(artistic=True, isvideo=False)

    clipa_rgb = _deoldify(clip, colorizer, render_factor, scenechange)

    return clipa_rgb


def _deoldify(clip: vs.VideoNode, colorizer: ModelImageVisualizer = None, render_factor: int = 24,
              scenechange: bool = True) -> vs.VideoNode:
    def deoldify_colorize(n: int, f: vs.VideoFrame, colorizer: ModelImageVisualizer = None,
                          render_factor: int = 24, scenechange: bool = True) -> vs.VideoFrame:

        if scenechange:
            is_scenechange = (n == 0) or (f.props['_SceneChangePrev'] == 1 and f.props['_SceneChangeNext'] == 0)
            if not is_scenechange:
                return f.copy()

        img_orig = frame_to_image(f)

        img_color = colorizer.get_transformed_image(img_orig, render_factor=render_factor, post_process=True)

        return image_to_frame(img_color, f.copy())

    return clip.std.ModifyFrame(clips=[clip],
                                selector=partial(deoldify_colorize, colorizer=colorizer, render_factor=render_factor,
                                                 scenechange=scenechange))


"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
wrapper to function ddcolor() with tweak pre-process.
"""


def vs_ddcolor(clip: vs.VideoNode, method: int = 2, model: int = 1, render_factor: int = 24,
               tweaks_enabled: bool = False, tweaks: list = [0.0, 1.0, 1.0, False, 0.3, 0.6, 1.5, 0.5],
               enable_fp16: bool = True, device_index: int = 0, num_streams: int = 1) -> vs.VideoNode:
    return vs_sc_ddcolor(clip, method, model, render_factor, tweaks_enabled, tweaks, enable_fp16, scenechange=False,
                         device_index=device_index, num_streams=num_streams)


def vs_sc_ddcolor(clip: vs.VideoNode, method: int = 2, model: int = 1, render_factor: int = 24,
                  tweaks_enabled: bool = False, tweaks: list = [0.0, 1.0, 1.0, False, 0.3, 0.6, 1.5, 0.5],
                  enable_fp16: bool = True, scenechange: bool = True, device_index: int = 0,
                  num_streams: int = 1) -> vs.VideoNode:
    if method == 0:
        return None
    else:
        import vsddcolor

    # input size must a multiple of 32
    input_size = math.trunc(render_factor / 2) * 32

    # unpack tweaks
    bright = tweaks[0]
    cont = tweaks[1]
    gamma = tweaks[2]
    luma_constrained_tweak = tweaks[3]
    luma_min = tweaks[4]
    gamma_luma_min = tweaks[5]
    gamma_alpha = tweaks[6]
    gamma_min = tweaks[7]
    if (len(tweaks) > 8):
        hue_adjust = tweaks[8].lower()
    else:
        hue_adjust = 'none'

    if tweaks_enabled:
        if luma_constrained_tweak:
            clipb = vs_sc_tweak(clip, bright=bright, cont=cont,
                                scenechange=scenechange)  # contrast and bright are adjusted before the constrainded luma and gamma
            clipb = sc_constrained_tweak(clipb, luma_min=luma_min, gamma=gamma, gamma_luma_min=gamma_luma_min,
                                         gamma_alpha=gamma_alpha, gamma_min=gamma_min, scenechange=scenechange)
        else:
            clipb = vs_sc_tweak(clip, bright=bright, cont=cont, gamma=gamma, scenechange=scenechange)
    else:
        clipb = clip

    # adjusting clip's color space to RGBH for vsDDColor
    if enable_fp16:
        clipb = vsddcolor.ddcolor(clipb.resize.Bicubic(format=vs.RGBH, range_s="full"), model=model,
                                  input_size=input_size, scenechange=scenechange, device_index=device_index,
                                  num_streams=num_streams)
    else:
        clipb = vsddcolor.ddcolor(clipb.resize.Bicubic(format=vs.RGBS, range_s="full"), model=model,
                                  input_size=input_size, scenechange=scenechange, device_index=device_index,
                                  num_streams=num_streams)

    # adjusting color space to RGB24 for deoldify
    clipb_rgb = clipb.resize.Bicubic(format=vs.RGB24, range_s="full")

    if tweaks_enabled and hue_adjust != 'none':
        clipb_rgb = vs_sc_adjust_clip_hue(clipb_rgb, hue_adjust, scenechange=scenechange)

    if tweaks_enabled:
        return vs_recover_clip_luma(clip, clipb_rgb)
    else:
        return clipb_rgb
