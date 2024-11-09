"""
-------------------------------------------------------------------------------
Author: Dan64
Date: 2024-09-14
version:
LastEditors: Dan64
LastEditTime: 2024-10-08
-------------------------------------------------------------------------------
Description:
-------------------------------------------------------------------------------
main Vapoursynth wrapper for model: "ColorMNet"
URL: https://github.com/yyang181/colormnet
Based on paper:
@inproceedings{yang2024colormnet,
    author = {Yang, Yixin and Dong, Jiangxin and Tang, Jinhui and Pan Jinshan},
    title = {ColorMNet: A Memory-based Deep Spatial-Temporal Feature Propagation Network for Video Colorization},
    booktitle = {ECCV},
    year = {2024}
}
"""
from __future__ import annotations, print_function
import threading
import torch
import os
import gc
from functools import partial

import vapoursynth as vs
from vsdeoldify.colormnet.colormnet_render import ColorMNetRender
from vsdeoldify.colormnet.colormnet_utils import *
from vsdeoldify.colormnet.colormnet_server import ColorMNetServer
from vsdeoldify.colormnet.colormnet_client import ColorMNetClient
from vsdeoldify.vsslib.imfilters import image_weighted_merge
from vsdeoldify.vsslib.vsutils import MessageType, HAVC_LogMessage

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_MODULE_LOADING"] = "LAZY"

# torch.cuda.set_device(0)

# import warnings

package_dir = os.path.dirname(os.path.realpath(__file__))


def vs_colormnet_local(clip: vs.VideoNode, clip_ref: vs.VideoNode, clip_sc: vs.VideoNode, image_size: int = -1,
                       enable_resize: bool = False, frame_propagate: bool = False, render_vivid: bool = True,
                       max_memory_frames: int = 0, ref_weight: float = 1.0) -> vs.VideoNode:
    vid_length = clip.num_frames

    colorizer = ColorMNetRender(image_size=image_size, vid_length=vid_length, enable_resize=enable_resize,
                                encode_mode=1, max_memory_frames=max_memory_frames, reset_on_ref_update=render_vivid,
                                project_dir=package_dir)

    clip_colored = _colormnet_async(colorizer, clip, clip_ref, clip_sc, frame_propagate, ref_weight)

    return clip_colored


def _colormnet_async(colorizer: ColorMNetRender, clip: vs.VideoNode, clip_ref: vs.VideoNode, clip_sc: vs.VideoNode,
                     frame_propagate: bool = False, ref_weight: float = 1.0) -> vs.VideoNode:
    def colormnet_clip_color_merge(n, f, colorizer: ColorMNetRender = None, propagate: bool = False,
                                   weight: float = 1.0) -> vs.VideoFrame:

        is_scenechange = f[2].props['_SceneChangePrev'] == 1
        is_scenechange_ext = is_scenechange and f[2].props['_SceneChangeNext'] == 1
        img_orig = frm_to_img(f[0])
        img_ref = frm_to_img(f[1])

        if n == 0:
            # vs.core.log_message(2, "Reference Frame: " + str(n))
            colorizer.set_ref_frame(img_ref)
        elif is_scenechange:
            frame_as_video = not is_scenechange_ext and propagate
            # vs.core.log_message(2, "Reference Frame: " + str(n))
            colorizer.set_ref_frame(img_ref, frame_as_video)
        else:
            colorizer.set_ref_frame(None)

        img_color = colorizer.colorize_frame(ti=n, frame_i=img_orig)

        # the frames that are not scenechange are merged with the ref frames generated by HAVC
        # this should stabilize further the colors generated with HAVC.
        if not is_scenechange:
            img_color_m = image_weighted_merge(img_color, img_ref, weight)
        else:
            img_color_m = img_color

        return img_to_frm(img_color_m, f[0].copy())

    def colormnet_clip_color(n, f, colorizer: ColorMNetRender = None, propagate: bool = False) -> vs.VideoFrame:

        is_scenechange = f[1].props['_SceneChangePrev'] == 1
        is_scenechange_ext = f[1].props['_SceneChangeNext'] == 1
        img_orig = frm_to_img(f[0])

        if n == 0:
            img_ref = frm_to_img(f[1])
            # vs.core.log_message(2, "Reference Frame: " + str(n))
            colorizer.set_ref_frame(img_ref)
        elif is_scenechange:
            img_ref = frm_to_img(f[1])
            # vs.core.log_message(2, "Reference Frame: " + str(n))
            frame_as_video = not is_scenechange_ext and propagate
            colorizer.set_ref_frame(img_ref, frame_as_video)
        else:
            colorizer.set_ref_frame(None)

        img_color = colorizer.colorize_frame(ti=n, frame_i=img_orig)

        return img_to_frm(img_color, f[0].copy())

    if 0 < ref_weight < 1 and not (clip_sc is None):
        clip_colored = clip.std.ModifyFrame(clips=[clip, clip_ref, clip_sc],
                                            selector=partial(colormnet_clip_color_merge, colorizer=colorizer,
                                                             propagate=frame_propagate, weight=ref_weight))
    else:
        clip_colored = clip.std.ModifyFrame(clips=[clip, clip_ref],
                                        selector=partial(colormnet_clip_color, colorizer=colorizer,
                                                         propagate=frame_propagate))
    return clip_colored


def vs_colormnet_remote(clip: vs.VideoNode, clip_ref: vs.VideoNode, clip_sc: vs.VideoNode, image_size: int = -1,
                        enable_resize: bool = False, frame_propagate: bool = False, render_vivid: bool = True,
                        max_memory_frames: int = 0, ref_weight: float = 1.0, server_port: int = 0) -> vs.VideoNode:
    vid_length = clip.num_frames

    server = ColorMNetServer(server_port=server_port).run_server()
    colorizer = ColorMNetClient(image_size=image_size, vid_length=vid_length, enable_resize=enable_resize,
                                encode_mode=0, max_memory_frames=max_memory_frames, reset_on_ref_update=render_vivid,
                                server_port=server.get_port())

    if not colorizer.is_initialized():
        HAVC_LogMessage(MessageType.EXCEPTION, "Failed to initialize ColorMNet[remote] try ColorMNet[local]")

    clip_colored = _colormnet_client(colorizer, clip, clip_ref, clip_sc, frame_propagate, ref_weight)

    return clip_colored


def _colormnet_client(colorizer: ColorMNetClient, clip: vs.VideoNode, clip_ref: vs.VideoNode, clip_sc: vs.VideoNode,
                      frame_propagate: bool = False, ref_weight: float = 1.0,) -> vs.VideoNode:
    def colormnet_client_color_merge(n, f, colorizer: ColorMNetClient = None, propagate: bool = False,
                                     weight: float = 1.0) -> vs.VideoFrame:

        is_scenechange = f[2].props['_SceneChangePrev'] == 1
        is_scenechange_ext = is_scenechange and f[2].props['_SceneChangeNext'] == 1
        img_orig = frm_to_img(f[0])
        img_ref = frm_to_img(f[1])

        if n == 0:
            # vs.core.log_message(2, "Reference Frame: " + str(n))
            colorizer.set_ref_frame(img_ref)
        elif is_scenechange:
            frame_as_video = not is_scenechange_ext and propagate
            # vs.core.log_message(2, "Reference Frame: " + str(n))
            colorizer.set_ref_frame(img_ref, frame_as_video)
        else:
            colorizer.set_ref_frame(None)

        img_color = colorizer.colorize_frame(ti=n, frame_i=img_orig)

        # the frames that are not scenechange are merged with the ref frames generated by HAVC
        # this should stabilize further the colors generated with HAVC.
        if not is_scenechange:
            img_color_m = image_weighted_merge(img_color, img_ref, weight)
        else:
            img_color_m = img_color

        return img_to_frm(img_color_m, f[0].copy())

    def colormnet_client_color(n, f, colorizer: ColorMNetClient = None, propagate: bool = False) -> vs.VideoFrame:

        is_scenechange = f[1].props['_SceneChangePrev'] == 1
        is_scenechange_ext = f[1].props['_SceneChangeNext'] == 1
        img_orig = frm_to_img(f[0])

        if n == 0:
            img_ref = frm_to_img(f[1])
            # vs.core.log_message(2, "Reference Frame: " + str(n))
            colorizer.set_ref_frame(img_ref)
        elif is_scenechange:
            img_ref = frm_to_img(f[1])
            # vs.core.log_message(2, "Reference Frame: " + str(n))
            frame_as_video = not is_scenechange_ext and propagate
            colorizer.set_ref_frame(img_ref, frame_as_video)
        else:
            colorizer.set_ref_frame(None)

        img_color = colorizer.colorize_frame(ti=n, frame_i=img_orig)

        return img_to_frm(img_color, f[0].copy())

    if 0 < ref_weight < 1 and not (clip_sc is None):
        clip_colored = clip.std.ModifyFrame(clips=[clip, clip_ref, clip_sc],
                                            selector=partial(colormnet_client_color_merge, colorizer=colorizer,
                                                             propagate=frame_propagate, weight=ref_weight))
    else:
        clip_colored = clip.std.ModifyFrame(clips=[clip, clip_ref],
                                            selector=partial(colormnet_client_color, colorizer=colorizer,
                                                             propagate=frame_propagate))
    return clip_colored


def _img_colormnet(img_color: Image, clip: vs.VideoNode) -> vs.VideoNode:
    def colormnet_clip_color(n, f, img: Image = None) -> vs.VideoFrame:
        return img_to_frm(img, f.copy())

    clip_colored = clip.std.ModifyFrame(clips=[clip],
                                        selector=partial(colormnet_clip_color, img=img_color))

    return clip_colored
