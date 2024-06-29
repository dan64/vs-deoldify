"""
------------------------------------------------------------------------------- 
Author: Dan64
Date: 2024-04-08
version: 
LastEditors: Dan64
LastEditTime: 2024-06-10
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
Library of Vapoursynth utility functions.
"""

import vapoursynth as vs
import os
import math
import numpy as np
import cv2
from PIL import Image
from functools import partial

IMG_EXTENSIONS = ['.png', '.PNG', '.jpg', '.JPG', '.jpeg', '.JPEG',
                  '.ppm', '.PPM', '.bmp', '.BMP']

"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
function to convert a VideoFrame in Pillow image 
(why not available in Vapoursynth ?) 
"""


def frame_to_image(frame: vs.VideoFrame) -> Image:
    npArray = np.dstack([np.asarray(frame[plane]) for plane in range(frame.format.num_planes)])
    return Image.fromarray(npArray, 'RGB')


"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
function to convert a VideoFrame in Pillow image 
(why not available in Vapoursynth ?) 
"""


def frame_to_np_array(frame: vs.VideoFrame) -> np.ndarray:
    npArray = np.dstack([np.asarray(frame[plane]) for plane in range(frame.format.num_planes)])
    return npArray


"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
function to convert a Pillow image in VideoFrame 
(why not available in Vapoursynth ?) 
"""


def image_to_frame(img: Image, frame: vs.VideoFrame) -> vs.VideoFrame:
    npArray = np.array(img)
    [np.copyto(np.asarray(frame[plane]), npArray[:, :, plane]) for plane in range(frame.format.num_planes)]
    return frame


"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
function to convert a np.array() image in VideoFrame 
"""


def np_array_to_frame(npArray: np.ndarray, frame: vs.VideoFrame) -> vs.VideoFrame:
    [np.copyto(np.asarray(frame[plane]), npArray[:, :, plane]) for plane in range(frame.format.num_planes)]
    return frame


"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
wrapper to function misc.SCDetect()
requires the dll: Hybrid/64bit/vsfilters/MiscFilter/MiscFilters/MiscFilters.dll
"""


def SceneDetect(clip: vs.VideoNode, threshold: float = 0.1, frequency: int = 0) -> vs.VideoNode:

    clip = clip.std.SetFrameProp(prop="sc_threshold", floatval=threshold)
    clip = clip.std.SetFrameProp(prop="sc_frequency", intval=frequency)

    if threshold == 0 and frequency == 0:
        return clip

    def set_scene_change_all(n, f) -> vs.VideoFrame:

        f_out = f.copy()

        f_out.props['_SceneChangePrev'] = 1
        f_out.props['_SceneChangeNext'] = 0

        return f_out

    if frequency == 1:
        return clip.std.ModifyFrame(clips=[clip], selector=partial(set_scene_change_all))

    sc = clip.resize.Point(format=vs.GRAY8, matrix_s='709')

    try:
        sc = sc.misc.SCDetect(threshold=threshold)
    except Exception as error:
        raise vs.Error("HAVC_ddeoldify: plugin 'MiscFilters.dll' not properly loaded/installed: -> " + str(error))

    def set_scene_change(n, f, freq: int = 0) -> vs.VideoFrame:

        f_out = f[0].copy()

        is_scenechange = (n == 0) or (f[1].props['_SceneChangePrev'] == 1 and f[1].props['_SceneChangeNext'] == 0)

        if freq > 1:
            is_scenechange = is_scenechange or (n % freq == 0)

        if is_scenechange:
            # vs.core.log_message(2, "SceneDetect n= " + str(n))
            f_out.props['_SceneChangePrev'] = 1
            f_out.props['_SceneChangeNext'] = 0
        else:
            f_out.props['_SceneChangePrev'] = 0
            f_out.props['_SceneChangeNext'] = 0

        return f_out

    sc = clip.std.ModifyFrame(clips=[clip, sc], selector=partial(set_scene_change, freq=frequency))

    return sc


def SceneDetectFromDir(clip: vs.VideoNode, sc_framedir: str = None, merge_ref_frame: bool = False,
                       ref_frame_ext: bool = True) -> vs.VideoNode:
    ref_list = get_ref_names(sc_framedir)

    if len(ref_list) == 0:
        raise vs.Error(
            f"HAVC_deepex: no reference frames found in '{sc_framedir}', allowed format is: ref_nnnnnn.[png|jpg]")

    ref_num_list = [get_ref_num(f) for f in ref_list]
    ref_num_list.sort()

    def set_scenechange(n: int, f: vs.VideoFrame, ref_num_list: list = None) -> vs.VideoFrame:

        fout = f.copy()

        if n in ref_num_list:
            fout.props['_SceneChangePrev'] = 1
            if ref_frame_ext:
                fout.props['_SceneChangeNext'] = 1
            else:
                fout.props['_SceneChangeNext'] = 0
        else:
            if merge_ref_frame:
                fout.props['_SceneChangePrev'] = f.props['_SceneChangePrev']
                fout.props['_SceneChangeNext'] = f.props['_SceneChangeNext']
            else:
                fout.props['_SceneChangePrev'] = 0
                fout.props['_SceneChangeNext'] = 0
        return fout

    sc = clip.std.ModifyFrame(clips=[clip], selector=partial(set_scenechange, ref_num_list=ref_num_list))

    return sc


def get_sc_props(clip: vs.VideoNode) -> tuple[float, int]:
    sc_threshold = 0
    sc_frequency = 0

    try:
        frame = clip.get_frame(0)
        sc_threshold = frame.props['sc_threshold']
        sc_frequency = frame.props['sc_frequency']
    except Exception as error:
        vs.core.log_message(2, "HAVC properties: 'sc_threshold', 'sc_frequency' not found in clip")

    return sc_threshold, sc_frequency


def CopySCDetect(clip: vs.VideoNode, sc: vs.VideoNode) -> vs.VideoNode:
    def copy_property(n, f) -> vs.VideoFrame:
        fout = f[0].copy()
        fout.props['_SceneChangePrev'] = f[1].props['_SceneChangePrev']
        fout.props['_SceneChangeNext'] = f[1].props['_SceneChangeNext']
        fout.props['sc_threshold'] = f[1].props['sc_threshold']
        fout.props['sc_frequency'] = f[1].props['sc_frequency']
        return fout

    return clip.std.ModifyFrame(clips=[clip, sc], selector=copy_property)


"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
Function to save the reference frames of a clip 
"""


def vs_sc_export_frames(clip: vs.VideoNode = None, sc_framedir: str = None, ref_offset: int = 0,
                        ref_ext: str = 'png', ref_override: bool = True) -> vs.VideoNode:

    def save_sc_frame(n, f, sc_framedir: str = None, ref_offset: int = 0, ref_ext: str = 'png', ref_override: bool = True):
        is_scenechange = (n == 0) or (f.props['_SceneChangePrev'] == 1 and f.props['_SceneChangeNext'] == 0)

        if is_scenechange:
            ref_n = n + ref_offset
            img = frame_to_image(f)
            img_path = os.path.join(sc_framedir, f"ref_{ref_n:06d}.{ref_ext}")
            if not ref_override and os.path.exists(img_path):
                return f.copy()  # do nothing
            img.save(img_path)

        return f.copy()

    clip = clip.std.ModifyFrame(clips=[clip], selector=partial(save_sc_frame, sc_framedir=sc_framedir,
                                ref_offset=ref_offset, ref_ext=ref_ext, ref_override=ref_override))

    return clip


def get_ref_num(filename: str = ""):
    fname = filename.split(".")[0]
    fnum = int(fname.split("_")[-1])
    return fnum


def get_ref_images(dir="./") -> list:
    img_ref_file = [os.path.join(dir, f) for f in os.listdir(dir) if is_ref_file(dir, f)]
    return img_ref_file


def get_ref_names(dir="./") -> list:
    img_ref_list = [f for f in os.listdir(dir) if is_ref_file(dir, f)]
    return img_ref_list


def is_ref_file(dir="./", fname: str = "") -> bool:
    filename = os.path.join(dir, fname)

    if not os.path.isfile(filename):
        return False

    return fname.startswith("ref_") and any(fname.endswith(extension) for extension in IMG_EXTENSIONS)
