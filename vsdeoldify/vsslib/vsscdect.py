"""
------------------------------------------------------------------------------- 
Author: Dan64
Date: 2024-04-08
version: 
LastEditors: Dan64
LastEditTime: 2024-10-08
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
from skimage.metrics import structural_similarity
from enum import IntEnum
import vsdeoldify.vsslib.vsutils as vsutil

"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
wrapper to function misc.SCDetect() (requires the dll: MiscFilters.dll)
if sc_tht_filter > 0 it will be activated the post change scene detection filter
based on SSIM. This metric is used to measure how similar are two images. 
It measures images luminance, contrast and structure and compare those values on 2 images.
Suggested values to use this features are:
threshold=0.03 (very sensitive threshold)
frequency=25 (at least a frame every 25 is selected)
sc_tht_filter=0.60  
"""


def SceneDetect(clip: vs.VideoNode, threshold: float = 0.10, frequency: int = 0, sc_tht_filter: float = 0,
                min_length: int = 1, tht_white: float = 0.95, tht_black: float = 0.05, frame_norm: bool = False,
                sc_debug: bool = False) -> vs.VideoNode:
    scdect = SceneDetection(sc_debug)

    return scdect.SceneDetect(clip, threshold, frequency, sc_tht_filter, min_length, tht_white, tht_black,
                              frame_norm)


def sc_clip_normalize(sc: vs.VideoNode, tht_white: float = 0.95, tht_black: float = 0.05) -> vs.VideoNode:
    def set_normalize(n, f, tht_white: float = 0.95, tht_black: float = 0.05) -> vs.VideoFrame:
        frame_np = vsutil.frame_to_np_array(f)

        frame_m = vsutil.frame_normalize(frame_np, tht_black, tht_white)

        return vsutil.np_array_to_frame(frame_m, f.copy())

    sc = sc.std.ModifyFrame(clips=[sc], selector=partial(set_normalize, tht_white=tht_white, tht_black=tht_black))

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


def SceneDetectFromDir(clip: vs.VideoNode, sc_framedir: str = None, merge_ref_frame: bool = False,
                       ref_frame_ext: bool = True) -> vs.VideoNode:
    ref_list = vsutil.get_ref_names(sc_framedir)

    if len(ref_list) == 0:
        raise vs.Error(
            f"HAVC_deepex: no reference frames found in '{sc_framedir}', allowed format is: ref_nnnnnn.[png|jpg]")

    ref_num_list = [vsutil.get_ref_num(f) for f in ref_list]
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


class SceneDetection:
    _sc_debug: bool = None
    # GLOBAL VARIABLES for _scene_detect_filter_task()
    _sc_last_index = None
    _sc_prev_y = None

    def __init__(self, sc_debug: bool = False):
        self._sc_debug = sc_debug
        self._sc_last_index = None
        self._sc_prev_y = None

    def SceneDetect(self, clip: vs.VideoNode, threshold: float = 0.10, frequency: int = 0, sc_tht_filter: float = 0,
                    min_length: int = 1, tht_white: float = 0.95, tht_black: float = 0.05, frame_norm: bool = False
                    ) -> vs.VideoNode:
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
            if frame_norm:
                sc = sc_clip_normalize(sc, tht_white, tht_black)
            sc = sc.misc.SCDetect(threshold=threshold)
        except Exception as error:
            raise vs.Error("HAVC_ddeoldify: plugin 'MiscFilters.dll' not properly loaded/installed: -> " + str(error))

        def set_scene_change(n, f, freq: int = 0, tht_white: float = 0.90, tht_black: float = 0.10) -> vs.VideoFrame:

            f_out = f[0].copy()

            f_y = vsutil.frame_to_np_array(f[1])[:, :, 0]

            f_luma = round(np.mean(f_y) / 255.0, 2)

            is_scenechange = (n == 0) or (f[1].props['_SceneChangePrev'] == 1 and f[1].props['_SceneChangeNext'] == 0)

            if freq > 1:
                is_scenechange = is_scenechange or (n % freq == 0)

            if is_scenechange and n == 0:
                # vs.core.log_message(2, "SceneDetect n= " + str(n))
                f_out.props['_SceneChangePrev'] = 1
                f_out.props['_SceneChangeNext'] = 0
            elif is_scenechange and tht_black < f_luma < tht_white:
                f_out.props['_SceneChangePrev'] = 1
                f_out.props['_SceneChangeNext'] = 0
            else:
                f_out.props['_SceneChangePrev'] = 0
                f_out.props['_SceneChangeNext'] = 0

            return f_out

        sc = clip.std.ModifyFrame(clips=[clip, sc], selector=partial(set_scene_change, freq=frequency,
                                                                     tht_white=tht_white, tht_black=tht_black))

        if sc_tht_filter > 0.0:
            clip = clip.std.CopyFrameProps(prop_src=sc, props=['_SceneChangePrev', '_SceneChangeNext'])
            sc = self.SceneDetectFilter(clip=clip, threshold=sc_tht_filter, tht_white=tht_white, tht_black=tht_black,
                                        min_length=min_length)

        return sc

    def SceneDetectFilter(self, clip: vs.VideoNode, threshold: float = 0.55, tht_white: float = 0.95,
                          tht_black: float = 0.05, min_length: int = 1) -> vs.VideoNode:
        t_step = 5000  # batch size for the SSIM filter (to avoid buffer memory problems)
        clip_length = clip.num_frames

        clip_list = []

        for i in range(0, clip_length, t_step):
            t_start = i
            t_end = min(t_start + t_step, clip_length)
            clip_cut = clip[t_start:t_end]
            clip_i = self._scene_detect_filter_task(t_start, clip_cut, threshold, tht_white, tht_black, min_length)
            clip_list.append(clip_i)

        clip_sc = vs.core.std.Splice(clip_list)
        return clip_sc

    def _scene_detect_filter_task(self, t_start: int, clip: vs.VideoNode, threshold: float = 0.55,
                                  tht_white: float = 0.95, tht_black: float = 0.05, min_length: int = 1
                                  ) -> vs.VideoNode:
        def set_scenechange(n: int, f: vs.VideoFrame, t_start: int, clip: vs.VideoNode, threshold: float = 0.55,
                            tht_white: float = 0.95, tht_black: float = 0.05, min_length: int = 1
                            ) -> vs.VideoFrame:
            fout = f.copy()

            np_img = cv2.cvtColor(vsutil.frame_to_np_array(f), cv2.COLOR_RGB2GRAY)
            luma = round(np.mean(np.array(np_img)) / 255, 3)
            y_last = np_img

            if n == 0:
                self._sc_last_index = None

            is_scenechange = fout.props['_SceneChangePrev'] == 1 or n == 0

            if is_scenechange and self._sc_last_index is None:
                fout.props['_SceneChangePrev'] = 1
                fout.props['_SceneChangeNext'] = 0
                self._sc_last_index = n
                self._sc_prev_y = y_last
                return fout

            if not is_scenechange:
                return fout

            if is_scenechange and (n > 0 and (n - self._sc_last_index) < min_length):
                fout.props['_SceneChangePrev'] = 0
                fout.props['_SceneChangeNext'] = 0
                return fout

            if n < clip.num_frames:
                score = structural_similarity(y_last, self._sc_prev_y, full=False)
                scene_change = (score < threshold and tht_black < luma < tht_white)
            else:
                score = 1
                scene_change = False

            if scene_change:
                if self._sc_debug:
                    vsutil.HAVC_LogMessage(vsutil.MessageType.WARNING,
                                           "SceneChange-> frame: ", (t_start + n), ", sc_index: ", self._sc_last_index,
                                           ", diff= ", score, ", luma= ", luma)
                fout.props['_SceneChangePrev'] = 1
                fout.props['_SceneChangeNext'] = 0
                self._sc_last_index = n
                self._sc_prev_y = y_last
            else:
                fout.props['_SceneChangePrev'] = 0
                fout.props['_SceneChangeNext'] = 0

            return fout

        sc = clip.std.ModifyFrame(clips=[clip],
                                  selector=partial(set_scenechange, t_start=t_start, clip=clip, threshold=threshold,
                                                   tht_white=tht_white, tht_black=tht_black, min_length=min_length))

        return sc
