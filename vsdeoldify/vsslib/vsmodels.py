"""
------------------------------------------------------------------------------- 
Author: Dan64
Date: 2024-04-08
version: 
LastEditors: Dan64
LastEditTime: 2025-10-26
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
module containing the main functions to colorize the frames with deoldify() and ddcolor().
"""
import vapoursynth as vs
import math
import torch
from functools import partial

from vsdeoldify.deoldify.visualize import ModelImageInitializer, ModelImageVisualizer
#from vsdeoldify.deoldify.visualize import *
from vsdeoldify.remaster import vs_sc_remaster_colorize
from vsdeoldify.vsslib.imfilters import image_weighted_merge
from vsdeoldify.vsslib.vsfilters import vs_sc_tweak, sc_constrained_tweak, vs_sc_adjust_clip_hue, vs_recover_clip_luma
from vsdeoldify.vsslib.vsutils import frame_to_image, image_to_frame, frame_to_np_array, np_array_to_frame
from vsdeoldify.vsslib.vsutils import debug_ModifyFrame
from vsdeoldify.colormnet import vs_colormnet_remote, vs_colormnet_local
from vsdeoldify.colormnet2 import vs_colormnet2_remote, vs_colormnet2_local
from vsdeoldify.deepex import deepex_colorizer, ModelColorizer
from vsdeoldify.colorization import ModelColorization
from vsdeoldify.havc_utils import rgb_denoise, vs_auto_levels

from vsdeoldify.vsslib.constants import *

def vs_colormnet(clip: vs.VideoNode, clip_ref: vs.VideoNode, clip_sc: vs.VideoNode, image_size: int = -1,
                 enable_resize: bool = False, frame_propagate: bool = True, render_vivid: bool = True,
                 max_memory_frames: int = 0, encode_mode: int = 0, ref_weight: float = 1.0) -> vs.VideoNode:
    if encode_mode == 1:
        if max_memory_frames is None or max_memory_frames == 0:
            gpu_mem_free, gpu_mem_total = torch.cuda.mem_get_info()
            mem_tot_k = round(gpu_mem_total / 1024 / 1024 / 1024, 0)
            if mem_tot_k < 8.5:
                max_memory_frames = 4
            elif mem_tot_k < 12.5:
                max_memory_frames = 8
            elif mem_tot_k < 16.5:
                max_memory_frames = 18
            else:
                max_memory_frames = 25

    match encode_mode:
        case 0 | 2:
            return vs_colormnet_remote(clip, clip_ref, clip_sc, image_size, enable_resize, frame_propagate,
                                       render_vivid, max_memory_frames, ref_weight, use_all_refs=(encode_mode == 2))
        case 1 | 3:  # encode_mode = 3 is supported only for testing, given the memory limitation of this method
            return vs_colormnet_local(clip, clip_ref, clip_sc, image_size, enable_resize, frame_propagate,
                                      render_vivid, max_memory_frames, ref_weight, use_all_refs=(encode_mode == 3))
        case _:
            raise vs.Error("HAVC_deepex: unknown encode mode: " + str(encode_mode))


def vs_colormnet2(clip: vs.VideoNode, clip_ref: vs.VideoNode, clip_sc: vs.VideoNode, image_size: int = -1,
                 enable_resize: bool = False, frame_propagate: bool = True, render_vivid: bool = True,
                 max_memory_frames: int = 0, encode_mode: int = 0, ref_weight: float = 1.0) -> vs.VideoNode:
    if encode_mode == 1:
        if max_memory_frames is None or max_memory_frames == 0:
            gpu_mem_free, gpu_mem_total = torch.cuda.mem_get_info()
            mem_tot_k = round(gpu_mem_total / 1024 / 1024 / 1024, 0)
            if mem_tot_k < 8.5:
                max_memory_frames = 4
            elif mem_tot_k < 12.5:
                max_memory_frames = 8
            elif mem_tot_k < 16.5:
                max_memory_frames = 18
            else:
                max_memory_frames = 25

    match encode_mode:
        case 0 | 2:
            return vs_colormnet2_remote(clip, clip_ref, clip_sc, image_size, enable_resize, frame_propagate,
                                       render_vivid, max_memory_frames, ref_weight, use_all_refs=(encode_mode == 2))
        case 1 | 3:  # encode_mode = 3 is supported only for testing, given the memory limitation of this method
            return vs_colormnet2_local(clip, clip_ref, clip_sc, image_size, enable_resize, frame_propagate,
                                      render_vivid, max_memory_frames, ref_weight, use_all_refs=(encode_mode == 3))
        case _:
            raise vs.Error("HAVC_cmnet2: unknown encode mode: " + str(encode_mode))


def vs_deepex(clip: vs.VideoNode, clip_ref: vs.VideoNode, clip_sc: vs.VideoNode, image_size: list = [432, 768],
              enable_resize: bool = False, wls_filter_on: bool = True, render_vivid: bool = True,
              propagate: bool = True,
              ref_weight: float = 1.0) -> vs.VideoNode:
    colorizer = deepex_colorizer(image_size=image_size, enable_resize=enable_resize)

    def deepex_clip_color_merge(n, f, colorizer: ModelColorizer = None, wls_on: bool = True,
                                propagate: bool = True, render_vivid: bool = True,
                                weight: float = 1.0) -> vs.VideoFrame:

        is_scenechange = f[2].props['_SceneChangePrev'] == 1
        is_scenechange_ext = is_scenechange and f[2].props['_SceneChangeNext'] == 1
        img_orig = frame_to_image(f[0])
        img_ref = frame_to_image(f[1])

        if n == 0:
            colorizer.set_ref_frame(img_ref)
        elif is_scenechange:
            # vs.core.log_message(2, "Reference Frame: " + str(n))
            frame_as_video = not is_scenechange_ext and propagate
            colorizer.set_ref_frame(img_ref, frame_propagate=frame_as_video)

        img_color = colorizer.colorize_frame(img_orig, wls_filter_on=wls_on, render_vivid=render_vivid)

        # the frames that are not scenechange are merged with the ref frames generated by HAVC
        # this should stabilize further the colors generated with HAVC.
        if not is_scenechange:
            img_color_m = image_weighted_merge(img_color, img_ref, weight)
        else:  # the frame obtained from a reference should be already good is merged with low weight
            img_color_m = img_color   # image_weighted_merge(img_color, img_ref, 0.20)

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
            frame_as_video = not is_scenechange_ext and propagate
            colorizer.set_ref_frame(img_ref, frame_as_video)

        img_color = colorizer.colorize_frame(img_orig, wls_filter_on=wls_on, render_vivid=render_vivid)

        return image_to_frame(img_color, f[0].copy())

    if 0 < ref_weight < 1 and not (clip_sc is None):
        clip_colored = clip.std.ModifyFrame(clips=[clip, clip_ref, clip_sc],
                                            selector=partial(deepex_clip_color_merge, colorizer=colorizer,
                                                             wls_on=wls_filter_on, render_vivid=render_vivid,
                                                             propagate=propagate, weight=ref_weight))
    else:
        clip_colored = clip.std.ModifyFrame(clips=[clip, clip_ref],
                                            selector=partial(deepex_clip_color, colorizer=colorizer,
                                                             propagate=propagate,
                                                             wls_on=wls_filter_on, render_vivid=render_vivid))
    return clip_colored


"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
wrapper to DeepRemaster. 
"""


def vs_deepremaster(clip: vs.VideoNode, clip_ref: vs.VideoNode, clip_sc: vs.VideoNode, render_vivid: bool = True,
                    ref_weight: float = 1.0, ref_size: int = 256, frame_size: int = 320, memory_size: int = None,
                    ref_frequency: int = 0, device_index: int = 0) -> vs.VideoNode:
    if memory_size is None or memory_size == 0:
        memory_size = DEF_NUM_RF_FRAMES
    if memory_size < DEF_MIN_RF_FRAMES:
        memory_size = DEF_MIN_RF_FRAMES

    clip_colored = vs_sc_remaster_colorize(clip, clip_ref, clip_sc=clip_sc, length=DEF_BATCH_SIZE,
                                           render_vivid=render_vivid, ref_minedge=ref_size,
                                           frame_mindim=frame_size, merge_weight=ref_weight,
                                           ref_buffer_size=memory_size, ref_frequency=ref_frequency,
                                           device_index=device_index)

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
                   scenechange: bool = True, package_dir: str = "") -> vs.VideoNode | None:
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
            is_scenechange = (n == 0) or (f.props['_SceneChangePrev'] == 1)
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
wrapper to Colorization. 
"""


def vs_sc_colorization(clip: vs.VideoNode, colorizer_model: str = 'siggraph17',
                       scenechange: bool = True, frame_size:int = 256) -> vs.VideoNode:
    m_colorizer = ModelColorization(model=colorizer_model, use_gpu=True)
    f_size = frame_size # min(frame_size, 512)

    def colorization(n: int, f: vs.VideoFrame, colorizer: ModelColorization = None,
                     scflag: bool = True, f_size: int = 256) -> vs.VideoFrame:

        if scflag:
            is_scenechange = (n == 0) or (f.props['_SceneChangePrev'] == 1)
            if not is_scenechange:
                return f.copy()

        np_frame_orig = frame_to_np_array(f)

        np_frame_colored = colorizer.colorize_frame(np_frame_orig)

        return np_array_to_frame(np_frame_colored, f.copy())

    #clip_new = debug_ModifyFrame(f_start=0, f_end=500, clip=clip, clips=[clip],
    #                             selector=partial(colorization, colorizer=m_colorizer, scflag=scenechange))
    clip_new = clip.std.ModifyFrame(clips=[clip], selector=partial(colorization, colorizer=m_colorizer,
                                    scflag=scenechange, f_size=f_size))

    return clip_new


"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
wrapper to function ddcolor() with tweak pre-process.
"""

def vs_ddcolor(clip: vs.VideoNode, method: int = 2, model: int = 1, render_factor: int = 24,
               tweaks_flags: list[bool] = (False, False, False),
               tweaks: list = (DEF_TWEAK_p, "none"),
               enable_fp16: bool = True, device_index: int = 0, num_streams: int = 1) -> vs.VideoNode:
    return vs_sc_ddcolor(clip, method, model, render_factor, tweaks_flags, tweaks, enable_fp16, scenechange=False,
                         device_index=device_index, num_streams=num_streams)


def vs_sc_ddcolor(clip: vs.VideoNode, method: int = 2, model: int = 1, render_factor: int = 24,
                  tweaks_flags: list[bool] = (False, False, False),
                  tweaks: list = (DEF_TWEAK_p, "none"),
                  enable_fp16: bool = True, scenechange: bool = True, device_index: int = 0,
                  num_streams: int = 1) -> vs.VideoNode | None:
    if method == 0:
        return None

    if model in (0, 1):
        import vsddcolor

    # input size must a multiple of 32
    input_size = math.trunc(render_factor / 2) * 32

    # unpack tweaks
    tweaks_enabled = tweaks_flags[0]
    denoise_enabled = tweaks_flags[1]
    retinex_enabled = tweaks_flags[2]

    if len(tweaks) == 2:
        bright = tweaks[0][0]
        cont = tweaks[0][1]
        gamma = tweaks[0][2]
        luma_constrained_tweak = tweaks[0][3]
        luma_min = tweaks[0][4]
        gamma_luma_min = tweaks[0][5]
        gamma_alpha = tweaks[0][6]
        gamma_min = tweaks[0][7]
        hue_adjust = tweaks[1].lower()
    else:
        bright = tweaks[0]
        cont = tweaks[1]
        gamma = tweaks[2]
        luma_constrained_tweak = tweaks[3]
        luma_min = tweaks[4]
        gamma_luma_min = tweaks[5]
        gamma_alpha = tweaks[6]
        gamma_min = tweaks[7]
        if len(tweaks) > 8:
            hue_adjust = tweaks[8]
        else:
            hue_adjust = 'none'

    if tweaks_enabled:
        if retinex_enabled:
            clipb = vs_auto_levels(clip, mode='strong', method=5, luma_blend=True, range_tv=True)
        elif luma_constrained_tweak:
            clipb = vs_sc_tweak(clip, bright=bright, cont=cont,
                                scenechange=scenechange)  # contrast and bright are adjusted before the constrainded luma and gamma
            clipb = sc_constrained_tweak(clipb, luma_min=luma_min, gamma=gamma, gamma_luma_min=gamma_luma_min,
                                         gamma_alpha=gamma_alpha, gamma_min=gamma_min, scenechange=scenechange)
        else:
            clipb = vs_sc_tweak(clip, bright=bright, cont=cont, gamma=gamma, scenechange=scenechange)
    else:
        clipb = clip

    if model > 1:
        if model == 2:
            clipb_rgb = vs_sc_colorization(clipb, colorizer_model='siggraph17', scenechange=scenechange, frame_size=input_size)
        else:
            clipb_rgb = vs_sc_colorization(clipb, colorizer_model='eccv16', scenechange=scenechange, frame_size=input_size)
    else:
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

    if hue_adjust != 'none':
        clipb_rgb = vs_sc_adjust_clip_hue(clipb_rgb, hue_adjust, scenechange=scenechange)

    if denoise_enabled:  # remove rgb noise
        clipb_rgb = rgb_denoise(clipb_rgb, denoise_levels=[0.3, 0.2], rgb_factors=[0.98, 1.02, 1.0])

    if tweaks_enabled:
        return vs_recover_clip_luma(clip, clipb_rgb)
    else:
        return clipb_rgb
