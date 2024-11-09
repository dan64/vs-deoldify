"""
-------------------------------------------------------------------------------
Author: Dan64
Date: 2024-09-14
version:
LastEditors: Dan64
LastEditTime: 2024-09-22
-------------------------------------------------------------------------------
Description:
-------------------------------------------------------------------------------
Utility functions for the Vapoursynth wrapper of ColorMNet.
"""
import os
from os import path
import vapoursynth as vs
import numpy as np
from PIL import Image
import io
from vsdeoldify.colormnet.dataset.range_transform import inv_im_trans, inv_lll2rgb_trans
from skimage import color
import cv2


def image_to_byte_array(img: Image, img_format: str = "jpeg", img_quality: int = 95) -> bytes:
    # BytesIO is a file-like buffer stored in memory
    img_byte_array = io.BytesIO()
    # image.save expects a file-like as an argument
    if img_format in ("jpg", "jpeg"):
        img.save(img_byte_array, format=img_format, subsampling=0, quality=img_quality)
    else:  # "png"
        img.save(img_byte_array, format=img_format)
    # Turn the BytesIO object back into a bytes object
    return img_byte_array.getvalue()


def byte_array_to_image(img_byte_array: bytes) -> Image:
    stream = io.BytesIO(img_byte_array)
    img = Image.open(stream).convert('RGB')
    return img


def detach_to_cpu(x):
    return x.detach().cpu()


def tensor_to_np_float(image):
    image_np = image.numpy().astype('float32')
    return image_np


def lab2rgb_transform_PIL(mask):
    mask_d = detach_to_cpu(mask)
    mask_d = inv_lll2rgb_trans(mask_d)
    im = tensor_to_np_float(mask_d)

    if len(im.shape) == 3:
        im = im.transpose((1, 2, 0))
    else:
        im = im[:, :, None]

    im = color.lab2rgb(im)

    return im.clip(0, 1)


def img_weighted_merge(img1: Image, img2: Image, weight: float = 0.5) -> Image:
    img1_np = np.asarray(img1)
    img2_np = np.asarray(img2)

    img_new = np.copy(img1_np)

    img_m = np.multiply(img1_np, 1 - weight).clip(0, 255).astype(int) + np.multiply(img2_np, weight).clip(0,
                                                                                                          255).astype(
        int)
    img_new[:, :, 0] = img_m[:, :, 0]
    img_new[:, :, 1] = img_m[:, :, 1]
    img_new[:, :, 2] = img_m[:, :, 2]

    return Image.fromarray(img_new)


def frm_to_img(frame: vs.VideoFrame) -> Image:
    np_array = np.dstack([np.asarray(frame[plane]) for plane in range(frame.format.num_planes)])
    return Image.fromarray(np_array, 'RGB')


def img_to_frm(img: Image, frame: vs.VideoFrame) -> vs.VideoFrame:
    np_array = np.array(img)
    [np.copyto(np.asarray(frame[plane]), np_array[:, :, plane]) for plane in range(frame.format.num_planes)]
    return frame
