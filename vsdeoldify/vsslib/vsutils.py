"""
------------------------------------------------------------------------------- 
Author: Dan64
Date: 2024-04-08
version: 
LastEditors: Dan64
LastEditTime: 2025-01-05
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


def HAVC_LogMessage(message_type: MessageType = MessageType.INFORMATION, message_text: str = None):
    if message_type == MessageType.EXCEPTION:
        raise vs.Error(message_text)
    else:
        vs.core.log_message(int(message_type), message_text)


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
Function to save the reference frames of a clip 
"""


def vs_sc_export_frames(clip: vs.VideoNode = None, sc_framedir: str = None, ref_offset: int = 0,
                        ref_ext: str = 'png', ref_jpg_quality: int = 95, ref_override: bool = True,
                        prop_name: str = "_SceneChangePrev") -> vs.VideoNode:
    pil_ext = ref_ext.lower()

    def save_sc_frame(n, f, sc_framedir: str = None, ref_offset: int = 0,
                      ref_ext: str = 'png', ref_jpg_quality: int = 95, ref_override: bool = True):

        is_scenechange = (n == 0) or (f.props[prop_name] == 1)

        if is_scenechange:
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
                                                               ref_offset=ref_offset, ref_ext=pil_ext,
                                                               ref_jpg_quality=ref_jpg_quality,
                                                               ref_override=ref_override))

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
                      clips: list[vs.VideoNode] = None, selector: partial = None) -> vs.VideoNode:

    if len(clips) == 1:
        if f_start > 0:
            frame = clips[0].get_frame(0)
            selector(0, frame)
        for n in range(f_start, f_end):
            frame = clips[0].get_frame(n)
            selector(n, frame)
    else:
        if f_start > 0:
            frame = []
            for j in range(0, len(clips)):
                frame.append(clips[j].get_frame(0))
            selector(0, frame)
        for n in range(f_start, f_end):
            frame = []
            for j in range(0, len(clips)):
                frame.append(clips[j].get_frame(n))
            selector(n, frame)

    return clip
