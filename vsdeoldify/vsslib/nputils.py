"""
------------------------------------------------------------------------------- 
Author: Dan64
Date: 2024-02-29
version: 
LastEditors: Dan64
LastEditTime: 2025-09-17
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
Library of Numpy utlity functions.
"""
import math
import numpy as np
import cv2
from PIL import Image

"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
implementation of max() function on numpy array, beacuse this function 
is not available on the base library.  
"""


def array_max(a: np.ndarray, a_max: np.ndarray, dtype: np.dtype = np.int32) -> np.ndarray:
    return np.where(a > a_max, a_max, a).astype(dtype)


"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
implementation of min() function on numpy array, beacuse this function 
is not available on the base library.  
"""


def array_min(a: np.ndarray, a_min: np.ndarray, dtype: np.dtype = np.int32) -> np.ndarray:
    return np.where(a < a_min, a_min, a).astype(dtype)


"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
implementation of min(max()) function on numpy array.
"""


def array_max_min(a: np.ndarray, a_max: np.ndarray, a_min: np.ndarray, dtype: np.dtype = np.int32) -> np.ndarray:
    a_m = array_max(a, a_max, dtype)
    return array_min(a_m, a_min, dtype)


def array_min_max(a: np.ndarray, a_min: np.ndarray, a_max: np.ndarray, dtype: np.dtype = np.int32) -> np.ndarray:
    a_m = array_max(a, a_max, dtype)
    return array_min(a_m, a_min, dtype)


"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
convert an NP image to gray or B&W if threshold > 0 
"""


def np_rgb_to_gray(img_np: np.ndarray, threshold: float = 0) -> np.ndarray:
    R = img_np[:, :, 0]
    G = img_np[:, :, 1]
    B = img_np[:, :, 2]

    R = R * 0.299
    G = G * 0.587
    B = B * 0.114

    tresh = round(threshold * 255)

    luma_np = R + G + B
    luma_np = luma_np.clip(0, 255)

    gray_np = img_np.copy()

    for i in range(3):
        if threshold > 0:
            gray_np[:, :, i] = np.where(luma_np > tresh, 255, 0)
        else:
            gray_np[:, :, i] = luma_np

    return gray_np


def np_get_luma(img_np: np.ndarray) -> np.ndarray:
    R = img_np[:, :, 0]
    G = img_np[:, :, 1]
    B = img_np[:, :, 2]

    R = R * 0.299
    G = G * 0.587
    B = B * 0.114

    luma_np = R + G + B
    luma_np = luma_np.clip(0, 255)

    return luma_np


def w_np_rgb_to_gray(img_np: np.ndarray, dark_luma: float = 0, luma_white: float = 0.90,
                     as_weight: bool = True) -> np.ndarray:
    R = img_np[:, :, 0]
    G = img_np[:, :, 1]
    B = img_np[:, :, 2]

    R = R * 0.299
    G = G * 0.587
    B = B * 0.114

    luma_np = R + G + B
    luma_np = luma_np.clip(0, 255)

    gray_np = img_np.copy()

    if dark_luma > 0:
        gray_np = gray_np.astype(float)
        max_white = round(luma_white * 255)

        tresh = min(round(dark_luma * 255), max_white - 10)

        grad = round(1 / (max_white - tresh), 3)

        luma_grad = ((luma_np - tresh) * grad).astype(float)

        weighted_luma = array_min_max(luma_grad, 0.0, 1.0, np.float32)

        if as_weight:
            gray_np = gray_np.astype(float)
        else:
            weighted_luma = np.multiply(weighted_luma, 255).clip(0, 255).astype(int)

        for i in range(3):
            gray_np[:, :, i] = weighted_luma

    else:
        if as_weight:
            gray_np = gray_np.astype(float)
            luma_np = np.divide(luma_np, 255.0)
        for i in range(3):
            gray_np[:, :, i] = luma_np

    return gray_np


"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
merge image1 with image2 using the mask (white->img2, black->img1) 
"""


def np_image_mask_merge(img1_np: np.ndarray, img2_np: np.ndarray,
                        mask_np: np.ndarray, normalize: bool = True) -> np.ndarray:
    if normalize:
        mask_white = (mask_np / 255).astype(float)  # pass only white
        mask_black = (1 - mask_white).astype(float)  # pass only black
    else:
        mask_white = mask_np.astype(float)
        mask_black = (1 - mask_white).astype(float)

    img_np = img1_np.copy()

    img_m = img1_np * mask_black + img2_np * mask_white

    for i in range(3):
        img_np[:, :, i] = img_m[:, :, i].clip(0, 255).astype(int)

    return img_np


"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
numpy weighted merge of image1 with image2 using the mask (white->img2, black->img1) 
"""


def w_np_image_mask_merge(img1_np: np.ndarray, img2_np: np.ndarray,
                          mask_w_np: np.ndarray, normalize: bool = False) -> np.ndarray:

    if normalize:
        mask_white = (mask_w_np / 255).astype(float)  # pass only white
        mask_black = (1 - mask_white).astype(float)  # pass only black
    else:
        mask_white = mask_w_np.astype(float)
        mask_black = (1 - mask_white).astype(float)

    img_np = img1_np.copy()

    img_m = (np.multiply(img1_np, mask_black) + np.multiply(img2_np, mask_white))

    for i in range(3):
        img_np[:, :, i] = img_m[:, :, i].clip(0, 255).astype(int)

    return img_np


"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
numpy implementation of image merge on 3 planes, faster than vs.core.std.Merge()
"""


def np_weighted_merge(img1_np: np.ndarray, img2_np: np.ndarray, weight: float = 0.5) -> np.ndarray:
    img_new = np.copy(img1_np)

    img_m = (np.multiply(img1_np, 1 - weight) + np.multiply(img2_np, weight)).clip(0, 255).astype(int)
    img_new[:, :, 0] = img_m[:, :, 0]
    img_new[:, :, 1] = img_m[:, :, 1]
    img_new[:, :, 2] = img_m[:, :, 2]

    return img_new

def np_luma_blend(img_np: np.ndarray, img_new_np: np.ndarray, f_luma: float = 0.5, luma_limit: float = 0.6,
                     alpha: float = 0.95, min_w: float = 0.10, decay: float = 2.0) -> np.ndarray:

    # Luma merge
    if f_luma < luma_limit:
        bright_scale = pow(f_luma / luma_limit, decay)
        w = max(alpha * bright_scale, min_w)
        # img_m = img * (1.0 - w) + img_new * w
        img_m = np_weighted_merge(img_np, img_new_np, w)
    else:
        img_m = img_new_np

    return img_m


"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
Function to copy the chroma parametrs "U", "V", of "img_m" in "orig" 
"""


def chroma_np_post_process(img_np: np.ndarray, orig_np: np.ndarray) -> np.ndarray:
    img_yuv = cv2.cvtColor(img_np, cv2.COLOR_RGB2YUV)
    # copy the chroma parametrs "U", "V", of "img_m" in "orig" 
    orig_yuv = cv2.cvtColor(orig_np, cv2.COLOR_RGB2YUV)
    orig_copy = np.copy(orig_yuv)
    orig_copy[:, :, 1:3] = img_yuv[:, :, 1:3]
    return cv2.cvtColor(orig_copy, cv2.COLOR_YUV2RGB)


"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
Function to add hue correction in cv2 HSV color space. 
hue range [-360.+360], converted to [-180.+180] 
"""


def np_hue_add(hsv_s: np.ndarray = None, hue: float = 0):
    if hue == 0:
        return hsv_s

    hue_half = 0.5 * min(max(int(hue), -360), 360)

    hsv_s = hsv_s + hue_half
    hsv_s = np.where(hsv_s > 180, hsv_s - 180, hsv_s)
    hsv_s = np.where(hsv_s < 0, hsv_s + 180, hsv_s)

    return hsv_s


def np_image_gamma_contrast(np_img: np.ndarray = None, gamma: float = 1.0, cont: float = 1.0, perc: float = 5):
    if cont == 1.0 and gamma == 1.0:
        return np_img

    yuv = cv2.cvtColor(np_img, cv2.COLOR_RGB2YUV)

    y = yuv[:, :, 0]
    yuv_new = np.copy(yuv)

    if cont != 1:
        y_min = np.percentile(y, perc)
        y_max = np.percentile(y, 100 - perc)
        y_fix = np.clip(y, y_min, y_max)
        y_cont = ((y_fix - y_min) * cont / (y_max - y_min))

        y_cont = array_min_max(y_cont, 0, 1, np.float64) * 255

        y_new = y_cont.clip(0, 255).astype(int)
    else:
        y_new = y

    if gamma != 1:
        y_new = np.power(y_new / 255, 1 / gamma)
        y_new = np.multiply(y_new, 255).clip(0, 255).astype(int)

    yuv_new[:, :, 0] = y_new

    np_img_rgb = cv2.cvtColor(yuv_new, cv2.COLOR_YUV2RGB)

    return np_img_rgb


def isfloat(x) -> bool:
    try:
        n = float(x)
        return True
    except ValueError:
        return False
