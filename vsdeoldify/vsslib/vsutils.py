"""
------------------------------------------------------------------------------- 
Author: Dan64
Date: 2024-04-08
version: 
LastEditors: Dan64
LastEditTime: 2025-10-19
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
Library of Vapoursynth utility functions.
"""

import vapoursynth as vs
import os
import numpy as np
from PIL import Image
from enum import IntEnum
from functools import partial

IMG_EXTENSIONS = ['.png', '.PNG', '.jpg', '.JPG', '.jpeg', '.JPEG',
                  '.ppm', '.PPM', '.bmp', '.BMP']


class MessageType(IntEnum):
    DEBUG = vs.MESSAGE_TYPE_DEBUG,
    INFORMATION = vs.MESSAGE_TYPE_INFORMATION,
    WARNING = vs.MESSAGE_TYPE_WARNING,
    CRITICAL = vs.MESSAGE_TYPE_CRITICAL,
    FATAL = vs.MESSAGE_TYPE_FATAL  # also terminates the process, should generally not be used by normal filters
    EXCEPTION = 10  # raise a fatal exception that terminates the process


"""
def HAVC_LogMessage(message_type: MessageType = MessageType.INFORMATION, message_text: str = None):
    if message_type == MessageType.EXCEPTION:
        raise vs.Error(message_text)
    else:
        vs.core.log_message(int(message_type), message_text)
"""

def HAVC_LogMessage(message_type: MessageType = MessageType.INFORMATION, *args):
    message_text: str = ' '.join(map(str, args))
    if message_type == MessageType.EXCEPTION:
        raise vs.Error(message_text)
    else:
        vs.core.log_message(int(message_type), message_text)

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
Functions to save the reference frames of a clip 
"""

def _select_frames_by_list(clip: vs.VideoNode, frame_list: list[int]) -> vs.VideoNode:
    """
    Returns a new clip containing only the frames whose indices are in `frame_list`.
    The frames are output in the order they appear in `frame_list`.
    Frame indices must be within [0, clip.num_frames].
    """
    if not frame_list:
        raise ValueError("frame_list is empty")

    # Validate frame numbers
    max_frame = clip.num_frames - 1
    if any(n < 0 or n > max_frame for n in frame_list):
        raise ValueError("Frame numbers in list must be in range [0, clip.num_frames - 1]")

    # Create a list of single-frame clips
    selected_clips = [clip[n] for n in frame_list]

    # Splice them into a single clip
    return vs.core.std.Splice(selected_clips)

# global variable for sc counting
_sc_counter: int
_sc_list: list[int]


def vs_sc_export_frames(clip: vs.VideoNode = None, sc_framedir: str = None, ref_offset: int = 0,
                        ref_ext: str = 'png', ref_jpg_quality: int = 95, ref_override: bool = True,
                        prop_name: str = "_SceneChangePrev", sequence: bool = False) -> vs.VideoNode:
    pil_ext = ref_ext.lower()
    global _sc_counter
    _sc_counter = 0

    def save_sc_frame(n, f, sc_framedir: str = None, ref_offset: int = 0, prop_name: str = "_SceneChangePrev",
                      ref_ext: str = 'png', ref_jpg_quality: int = 95, ref_override: bool = True,
                      sequence: bool = False):
        global _sc_counter

        is_scenechange = (n == 0) or (f.props[prop_name] == 1)
        if is_scenechange:
            if sequence:
                ref_n = _sc_counter
                _sc_counter = _sc_counter + 1
            else:
                ref_n = n + ref_offset
            img = frame_to_image(f)
            img_path = os.path.join(sc_framedir, f"ref_{ref_n:06d}.{ref_ext}")
            if not ref_override and os.path.exists(img_path):
                return f.copy()  # do nothing
            if ref_ext == "jpg":
                img.save(img_path, subsampling=0, quality=ref_jpg_quality)
            else:
                img.save(img_path)

        return f.copy()

    clip = clip.std.ModifyFrame(clips=[clip], selector=partial(save_sc_frame, sc_framedir=sc_framedir,
                                                               ref_offset=ref_offset, prop_name=prop_name,
                                                               ref_ext=pil_ext, ref_jpg_quality=ref_jpg_quality,
                                                               ref_override=ref_override, sequence=sequence))

    return clip


def vs_list_export_frames(clip: vs.VideoNode = None, sc_framedir: str = None, ref_list: list[int] = None,
                          offset: int = 0, ref_ext: str = 'png', ref_jpg_quality: int = 95, ref_override: bool = True,
                          fast_extract: bool = True) -> vs.VideoNode:
    pil_ext = ref_ext.lower()

    if len(ref_list) == 1: # the list is automatically generated
        sorted_list = list(range(0, clip.num_frames, ref_list[0]))
    else: # the list is sorted and duplicate frames are removed
        sorted_list = sorted(set(ref_list))

    if offset > 0:
        sorted_list = [num + offset for num in sorted_list]
    
    def save_sc_frame(n, f, sc_framedir: str = None, ref_list: list[int] = None, ref_ext: str = 'png',
                      ref_jpg_quality: int = 95, ref_override: bool = True, fast_extract: bool = True):
        global _sc_counter

        if fast_extract:
            is_scenechange = True
            f_num = ref_list[n]
        else:
            is_scenechange = (n in ref_list)
            f_num = n
        if is_scenechange:
            img = frame_to_image(f)
            img_path = os.path.join(sc_framedir, f"ref_{f_num:06d}.{ref_ext}")
            if not ref_override and os.path.exists(img_path):
                return f.copy()  # do nothing
            if ref_ext == "jpg":
                img.save(img_path, subsampling=0, quality=ref_jpg_quality)
            else:
                img.save(img_path)

        return f.copy()

    if fast_extract:
        clip_ref = _select_frames_by_list(clip, sorted_list)
    else:
        clip_ref = clip

    clip_new = clip_ref.std.ModifyFrame(clips=[clip_ref], selector=partial(save_sc_frame, sc_framedir=sc_framedir,
                                        ref_list=sorted_list, ref_ext=pil_ext, ref_jpg_quality=ref_jpg_quality,
                                        ref_override=ref_override, fast_extract=fast_extract))

    #clip_new = debug_ModifyFrame(f_start=0, f_end=147, clip=clip_ref, clips=[clip_ref],
    #                             selector=partial(save_sc_frame, sc_framedir=sc_framedir,
    #                                    ref_list=sorted_list, ref_ext=pil_ext, ref_jpg_quality=ref_jpg_quality,
    #                                    ref_override=ref_override, fast_extract=fast_extract), silent=True)
    return clip_new


def vs_get_video_ref(clip: vs.VideoNode = None, prop_name: str = "_SceneChangePrev") -> vs.VideoNode:
    global _sc_list, _sc_counter
    _sc_list = []

    def get_sc_list(n, f, prop_name: str):
        global _sc_list

        is_scenechange = (n == 0) or (f.props[prop_name] == 1)
        if is_scenechange:
            _sc_list.append(n)
        return f.copy()

    clip = clip.std.ModifyFrame(clips=[clip], selector=partial(get_sc_list, prop_name=prop_name))

    # set property to set the next reference frame position
    clip = clip.std.SetFrameProp(prop="sc_next_frame", intval=0)
    _sc_counter = 0

    def set_sc_list(n, f, sc_list: list[int], prop_name: str):
        global _sc_counter
        f_out = f.copy()
        is_scenechange = (n == 0) or (f.props[prop_name] == 1)
        if is_scenechange:
            if _sc_counter < len(sc_list):
                f_out.props["sc_next_frame"] = sc_list[_sc_counter]
            else:
                f_out.props["sc_next_frame"] = -1   # end list
            _sc_counter = _sc_counter + 1
        else:
            f_out.props["sc_next_frame"] = 0

        return f.copy()

    clip = clip.std.ModifyFrame(clips=[clip], selector=partial(set_sc_list, sc_list=_sc_list, prop_name=prop_name))

    return clip


def get_ref_last_list() -> list[int]:
    global _sc_list
    return _sc_list


def get_ref_num(filename: str = ""):
    fname = filename.split(".")[0]
    fnum = int(fname.split("_")[-1])
    return fnum


def get_ref_images(in_dir="./") -> list:
    img_ref_file = [os.path.join(in_dir, f) for f in os.listdir(in_dir) if is_ref_file(in_dir, f)]
    return img_ref_file


def get_ref_names(in_dir="./") -> list:
    img_ref_list = [f for f in os.listdir(in_dir) if is_ref_file(in_dir, f)]
    return img_ref_list


def is_ref_file(in_dir="./", fname: str = "") -> bool:
    filename = os.path.join(in_dir, fname)

    if not os.path.isfile(filename):
        return False

    return fname.startswith("ref_") and any(fname.endswith(extension) for extension in IMG_EXTENSIONS)


def frame_normalize(frame_np: np.ndarray, tht_black: float = 0.10, tht_white: float = 0.90) -> np.ndarray:
    frame_y = frame_np[:, :, 0]

    frame_luma = np.mean(frame_y) / 255.0

    if frame_luma <= tht_black or frame_luma >= tht_white:
        return frame_np

    img_np = frame_np.copy()

    frame_y = np.multiply(255, (frame_y - np.min(frame_y)) / (np.max(frame_y) - np.min(frame_y)))

    img_np[:, :, 0] = frame_y.clip(0, 255).astype('uint8')

    return img_np


def mean_pixel_distance(y_left: np.ndarray, y_right: np.ndarray, normalize: bool = True) -> float:
    """Return the mean average distance in pixel values between `left` and `right`.
    Both `left and `right` should be 2-dimensional 8-bit images of the same shape.
    """

    if normalize:
        luma_left = int(np.mean(y_left))
        luma_right = int(np.mean(y_right))
        if luma_right > luma_left:
            y_left = (y_left + (luma_right - luma_left)).clip(0, 255).astype('uint8')
        else:
            y_right = (y_right - (luma_right - luma_left)).clip(0, 255).astype('uint8')

    num_pixels: float = float(y_left.shape[0] * y_left.shape[1])
    dist = np.sum(np.abs(y_left.astype(np.int32) - y_right.astype(np.int32))) / num_pixels
    return dist / 255.0


def debug_ModifyFrame(f_start: int = 0, f_end: int = 1, clip: vs.VideoNode = None,
                      clips: list[vs.VideoNode] = None, selector: partial = None, silent: bool = True) -> vs.VideoNode:
    f_end = min(f_end, clip.num_frames - 1)
    if len(clips) == 1:
        if f_start > 0:
            frame = clips[0].get_frame(0)
            if not silent:
                print("Debug Frame: ", 0)
            selector(0, frame)
        for n in range(f_start, f_end):
            frame = clips[0].get_frame(n)
            if not silent:
                print("Debug Frame: ", n)
            selector(n, frame)
    else:
        if f_start > 0:
            frame = []
            for j in range(0, len(clips)):
                frame.append(clips[j].get_frame(0))
            if not silent:
                print("Debug Frame: ", 0)
            selector(0, frame)
        for n in range(f_start, f_end):
            frame = []
            for j in range(0, len(clips)):
                frame.append(clips[j].get_frame(n))
            if not silent:
                print("Debug Frame: ", n)
            selector(n, frame)

    return clip
