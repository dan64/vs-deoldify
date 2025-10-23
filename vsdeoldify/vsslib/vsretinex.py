"""
------------------------------------------------------------------------------- 
Author: Dan64
Date: 2024-04-08
version: 
LastEditors: Dan64
LastEditTime: 2025-10-01
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
Utility function for Retinex.
"""

import vapoursynth as vs
import numpy as np
import cv2
from PIL import Image
from functools import partial
from vsdeoldify.vsslib.imfilters import get_image_luma, image_luma_blend
from vsdeoldify.vsslib.vsplugins import load_Retinex_plugin
from vsdeoldify.vsslib.vsutils import frame_to_image, image_to_frame
from vsdeoldify.vsslib.vsfilters import vs_recover_clip_luma


def vs_retinex(clip: vs.VideoNode, luma_dark: float = 0.20, luma_bright: float = 0.80,
               sigmas: list[float] = (25, 80, 250), range_tv_in: bool = True, range_tv_out: bool = True,
               blend: bool = False, fast_mode:bool = True) -> vs.VideoNode:
    orig_fmt_id = clip.format.id
    orig_fmt_family = clip.format.color_family

    if clip.format.id != vs.RGB24:
        range_s = "limited" if range_tv_in else "full"
        # clip not in RGB24 format, it will be converted
        if clip.format.color_family == vs.YUV:
            clip = clip.resize.Bicubic(format=vs.RGB24, matrix_in_s="709", range_s=range_s,
                                       dither_type="error_diffusion")
        else:
            clip = clip.resize.Bicubic(format=vs.RGB24, range_s=range_s)

    if fast_mode:
        clip_out = vs_retinex_fast(clip, luma_dark, luma_bright, sigmas, range_tv_in, range_tv_out, blend)
    else:
        clip_out = vs_retinex_slow(clip, luma_dark, luma_bright, sigmas, range_tv_in, range_tv_out, blend)

    if orig_fmt_id != vs.RGB24 and orig_fmt_family == vs.YUV:
        range_s = "limited" if range_tv_out else "full"
        clip_out = clip_out.resize.Bicubic(clip=clip_out, format=orig_fmt_id, matrix_s="709", range_in_s="full",
                                           range_s=range_s)

    return clip_out

def vs_retinex_fast(clip: vs.VideoNode, luma_dark: float = 0.20, luma_bright: float = 0.80,
               sigmas: list[float] = (25, 80, 250), range_tv_in: bool = True, range_tv_out: bool = True,
               blend: bool = False) -> vs.VideoNode:

    load_Retinex_plugin()

    try:
        clip_rtx = vs.core.retinex.MSRCP(input=clip, sigma=sigmas, lower_thr=0.001, upper_thr=0.001, fulls=range_tv_in,
                                         fulld=range_tv_out, chroma_protect=1.200)
    except Exception as error:
        raise vs.Error("vs_retinex: plugin 'Retinex.dll' not properly loaded/installed -> " + str(error))

    def filter_retinex(n, f, luma_dark: float, luma_bright: float, range_tv, blend: bool):

        img = frame_to_image(f[0])

        if range_tv:
            maxrange = 235
            f_luma = max(get_image_luma(img, maxrange) - 0.07, 0)
        else:
            maxrange = 255
            f_luma = get_image_luma(img, maxrange)

        f_luma_bright = luma_dark <= f_luma <= luma_bright

        if not f_luma_bright:
            return f[0].copy()

        img_new = frame_to_image(f[1])
        img_m = image_luma_blend(img, img_new, f_luma, 0.40, 0.90, 0.25, 3.0) if blend else img_new

        return image_to_frame(img_m, f[0].copy())

    clip_out = clip.std.ModifyFrame(clips=[clip, clip_rtx], selector=partial(filter_retinex, luma_dark=luma_dark,
                                    luma_bright=luma_bright, range_tv=range_tv_in, blend=blend))

    return clip_out

def vs_retinex_slow(clip: vs.VideoNode, luma_dark: float = 0.20, luma_bright: float = 0.80,
               sigmas: list[float] = (25, 80, 250), range_tv_in: bool = True, range_tv_out: bool = True,
               blend: bool = False, chroma_resize: bool = True) -> vs.VideoNode:

    if chroma_resize and clip.width > 384:
        frame_size = 384
        rgb_clip = clip.resize.Spline36(width=frame_size, height=frame_size)
    else:
        chroma_resize = False
        rgb_clip = clip

    def retinex_ssr(img: np.ndarray, sigma: float = 300.0) -> np.ndarray:
        img = img.astype(np.float64) + 1.0
        log_img = np.log(img)
        log_illum = np.log(cv2.GaussianBlur(img, (0, 0), sigma))
        # log_illum = np.log(cv2.blur(img, (0, 0), sigma))  # should be 2x faster
        return log_img - log_illum

    def retinex_msr(img: np.ndarray, sigmas: list[float] = (15, 80, 250)) -> np.ndarray:
        ret = np.zeros_like(img, dtype=np.float64)
        for sigma in sigmas:
            ret += retinex_ssr(img, sigma)
        return ret / len(sigmas)

    def frame_retinex_MSR(n, f, sigmas: list[float], luma_dark: float, luma_bright: float, blend: bool, range_tv: bool):

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

        f_luma_bright = luma_dark <= f_luma <= luma_bright

        if not f_luma_bright:
            # HAVC_LogMessage(MessageType.WARNING, "HAVC_bw_tune: frame ", n, " luma: ", f_luma)
            return f.copy()

        # Apply MSR on Y only
        y_enhanced = retinex_msr(y_image, sigmas=sigmas)  # Pass 2D y_image directly

        # Normalize Y to [minrange, maxrange]
        y_min, y_max = np.min(y_enhanced), np.max(y_enhanced)
        if y_max - y_min < 1e-6:
            y_norm = np.full_like(y_image, (minrange + maxrange) / 2)
        else:
            y_norm = (y_enhanced - y_min) / (y_max - y_min) * (maxrange - minrange) + minrange

        yuv[:, :, 0] = np.clip(y_norm, minrange, maxrange)
        enhanced_rgb = cv2.cvtColor(yuv.astype(np.uint8), cv2.COLOR_YUV2RGB)

        img_new = Image.fromarray(enhanced_rgb, 'RGB')

        img_m = image_luma_blend(img, img_new, f_luma, 0.40, 0.90, 0.15, 4.0) if blend else img_new

        return image_to_frame(img_m, f.copy())

    clip_out = rgb_clip.std.ModifyFrame(clips=rgb_clip, selector=partial(frame_retinex_MSR, sigmas=sigmas,
                                                        luma_dark=luma_dark, luma_bright=luma_bright, blend=blend,
                                                        range_tv=range_tv_in))

    if chroma_resize:
        clip_resized = clip_out.resize.Spline36(width=clip.width, height=clip.height)
        clip_out = vs_recover_clip_luma(clip, clip_resized)

    return clip_out