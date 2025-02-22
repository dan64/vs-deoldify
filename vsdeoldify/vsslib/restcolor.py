"""
------------------------------------------------------------------------------- 
Author: Dan64
Date: 2024-04-08
version: 
LastEditors: Dan64
LastEditTime: 2025-02-21
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
Library of functions used by "HAVC" to restore color and change the hue of frames.
"""
import vapoursynth as vs
import math
import numpy as np
import cv2
from PIL import Image

from .nputils import *

"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
Restore the colors of past/future frame. The restore is applied using a mask
to select only the gray images on HSV color space.
The ranges that OpenCV manage for HSV format are the following:
- Hue range is [-180,+180], 
- Saturation range is [0,255] 
- Value range is [0,255].
For the 8-bit images, H is converted to H/2 to fit to the [0,255] range. 
So the range of hue in the HSV color space of OpenCV is [0,179]
The vector order is: H = 0, S = 1, V = 2
"""


def restore_color(img_color: Image = None, img_gray: Image = None, sat: float = 1.0, tht: int = 15, weight: float = 0,
                  tht_scen: float = 0.8, hue_adjust: str = 'none', return_mask: bool = False) -> Image:
    np_color = np.asarray(img_color)
    np_gray = np.asarray(img_gray)

    hsv_color = cv2.cvtColor(np_color, cv2.COLOR_RGB2HSV)
    hsv_gray = cv2.cvtColor(np_gray, cv2.COLOR_RGB2HSV)

    # desatured the color image
    hsv_color[:, :, 1] = hsv_color[:, :, 1] * min(max(sat, 0), 10)

    np_color_sat = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2RGB)

    hsv_s = hsv_gray[:, :, 1]

    hsv_mask = np.where(hsv_s < tht, 255, 0)  # white only gray pixels

    scenechange = np.mean(hsv_mask) / 255

    if 0 < tht_scen < 1 and scenechange > tht_scen:
        if hue_adjust != "" and hue_adjust != "none":
            return adjust_hue_range(img_gray, hue_adjust=hue_adjust)
        else:
            return img_gray

    mask_rgb = np_gray.copy()

    for i in range(3):
        mask_rgb[:, :, i] = hsv_mask

    if return_mask:
        return Image.fromarray(mask_rgb, 'RGB').convert('RGB')

    np_restored = np_image_mask_merge(np_gray, np_color_sat, mask_rgb)

    if weight > 0:
        np_restored = np_weighted_merge(np_restored, np_gray, weight)  # merge with gray frame
    if weight < 0:
        np_restored = np_weighted_merge(np_restored, np_color_sat, -weight)  # merge with colored frame

    img_restored = Image.fromarray(np_restored, 'RGB').convert('RGB')

    if hue_adjust != "" and hue_adjust != "none":
        return adjust_hue_range(img_restored, hue_adjust=hue_adjust)
    else:
        return img_restored


"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
Restore the gray frame colors frame. The restore is applied using a gradient mask
to select only the gray images on HSV color space.
The vector order is: H = 0, S = 1, V = 2
"""


def restore_color_gradient(img_color: Image = None, img_gray: Image = None, sat: float = 1.0, tht: int = 15,
                           weight: float = 0, alpha: float = 2.0, return_mask: bool = False) -> Image:
    np_color = np.asarray(img_color)
    np_gray = np.asarray(img_gray)

    hsv_color = cv2.cvtColor(np_color, cv2.COLOR_RGB2HSV)
    hsv_gray = cv2.cvtColor(np_gray, cv2.COLOR_RGB2HSV)

    # desatured the color image
    if sat != 1.0:
        hsv_color[:, :, 1] = hsv_color[:, :, 1] * min(max(sat, 0), 10)

    np_color_sat = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2RGB)

    hsv_s = hsv_gray[:, :, 1]

    hsv_mask = w_np_gradient_mask(hsv_s, tht, alpha)  # white only gray pixels

    mask_rgb = np_gray.copy()

    for i in range(3):
        mask_rgb[:, :, i] = hsv_mask

    if return_mask:
        return Image.fromarray(mask_rgb, 'RGB').convert('RGB')

    np_restored = w_np_image_mask_merge(np_gray, np_color_sat, mask_rgb, normalize=True)

    if weight > 0:
        np_restored = np_weighted_merge(np_restored, np_color_sat, weight)  # merge with colored frame

    img_restored = Image.fromarray(np_restored, 'RGB').convert('RGB')

    return img_restored


def w_np_gradient_mask(img_np: np.ndarray, tht: int = 15, alpha: float = 2.0, steep: float = 2.0) -> np.ndarray:

    luma_np = img_np.clip(0, 255)

    # grad = np.where(luma_np < tht, luma_np, tht + (luma_np - tht)*alpha)
    # luma_grad = (255.0 - luma_np - grad).clip(0, 255).astype(int)

    grad = np.where(luma_np < tht, steep*luma_np/alpha - tht, steep*(luma_np - tht)*alpha)
    luma_grad = (255.0 - tht - grad).clip(0, 255).astype(int)

    return luma_grad

"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
Change a given range of colors in HSV color space. 
The range is defined by the hue values in degree (range: 0-360)
In OpenCV, for the 8-bit images, H is converted to H/2 to fit to the [0,255] range. 
So the range of hue in the HSV color space of OpenCV is [0,179].
hue_range syntax: "hue1_min:hue1_max,..,hueN_min,hueN_max|adjust, weight"
where:
adjust: if > 0 and < 10 -> saturation parameter else -> hue_shift
weight: if > 0 -> merge with desaturared frame, if < 0 -> merge with colored orginal frame
"""


def adjust_hue_range(img_color: Image = None, hue_adjust: str = 'none', return_mask: bool = False) -> Image:
    if hue_adjust == 'none' or hue_adjust == '':
        return img_color

    param = _parse_hue_adjust(hue_adjust)

    if param is None:
        return img_color

    hue_range = param[0]
    sat = param[1]
    hue = param[2]
    weight = param[3]

    return adjust_chroma(img_color=img_color, hue_range=hue_range, sat=sat, hue=hue, weight=weight,
                         return_mask=return_mask)


def adjust_chroma(img_color: Image = None, hue_range: str = 'none', sat: float = 0.3, hue: int = 0, weight: float = 0,
                  return_mask: bool = False) -> Image:
    if hue_range == 'none' or hue_range == '':
        return img_color

    np_color = np.asarray(img_color)

    np_gray = np_color.copy()
    np_gray = cv2.cvtColor(np_gray, cv2.COLOR_RGB2HSV)

    hsv_color = cv2.cvtColor(np_color, cv2.COLOR_RGB2HSV)

    # apply hue correction, range [-180,+180]
    if hue != 0:
        np_gray[:, :, 0] = np_hue_add(np_gray[:, :, 0], hue)

    # desatured the color image
    if sat != 1:
        np_gray[:, :, 1] = np_gray[:, :, 1] * min(max(sat, 0), 10)

    np_gray_rgb = cv2.cvtColor(np_gray, cv2.COLOR_HSV2RGB)

    hsv_s = hsv_color[:, :, 0]

    cond = _build_hue_conditions(hsv_s, hue_range)

    hsv_mask = np.where(cond, 255, 0)  # white only gray pixels

    mask_rgb = np_color.copy()

    for i in range(3):
        mask_rgb[:, :, i] = hsv_mask

    if return_mask:
        return Image.fromarray(mask_rgb, 'RGB').convert('RGB')

    np_restored = np_image_mask_merge(np_color, np_gray_rgb, mask_rgb)

    if weight > 0:
        if hue == 0:
            np_restored = np_weighted_merge(np_restored, np_gray_rgb, weight)
        else:
            np_restored = np_weighted_merge(np_restored, np_color, weight)
    if weight < 0:   # use np_color instead of np_gray_rgb, is assumed that hue == 0 (no color mapping)
        np_restored = np_weighted_merge(np_restored, np_color, -weight)

    return Image.fromarray(np_restored, 'RGB').convert('RGB')


def np_image_chroma_tweak(img_color_rgb: np.ndarray, sat: float = 1, bright: float = 0, hue: int = 0,
                          hue_adjust: str = 'none') -> np.ndarray:
    if sat == 1 and bright == 0 and hue == 0 and hue_adjust == 'none':
        return img_color_rgb  # non changes

    hsv = cv2.cvtColor(img_color_rgb, cv2.COLOR_RGB2HSV)

    hsv[:, :, 0] = np_hue_add(hsv[:, :, 0], hue)
    hsv[:, :, 1] = hsv[:, :, 1] * min(max(sat, 0), 10)
    hsv[:, :, 2] = hsv[:, :, 2] * min(max(1 + bright, 0), 10)

    np_color_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    if hue_adjust == 'none' or hue_adjust == '':
        return np_color_rgb

    param = _parse_hue_adjust(hue_adjust)

    if param is None:
        return np_color_rgb

    hue_range = param[0]
    sat = param[1]
    hue = param[2]  # override hue with the new value
    weight = param[3]

    np_gray = np_color_rgb.copy()
    np_gray = cv2.cvtColor(np_gray, cv2.COLOR_RGB2HSV)

    hsv_color = hsv.copy()

    # apply hue correction, range [-180.+180], converted to [-90.+90]
    if hue != 0:
        np_gray[:, :, 0] = np_hue_add(np_gray[:, :, 0], hue)

    # desatured the color image
    if sat != 1:
        np_gray[:, :, 1] = np_gray[:, :, 1] * min(max(sat, 0), 10)

    np_gray_rgb = cv2.cvtColor(np_gray, cv2.COLOR_HSV2RGB)

    hsv_s = hsv_color[:, :, 0]

    cond = _build_hue_conditions(hsv_s, hue_range)

    hsv_mask = np.where(cond, 255, 0)  # white only gray pixels

    mask_rgb = img_color_rgb.copy()

    for i in range(3):
        mask_rgb[:, :, i] = hsv_mask

    np_restored = np_image_mask_merge(img_color_rgb, np_gray_rgb, mask_rgb)

    if weight > 0:
        if hue == 0:
            np_restored = np_weighted_merge(np_restored, np_gray_rgb, weight)
        else:
            np_restored = np_weighted_merge(np_restored, img_color_rgb, weight)
    if weight < 0:
        np_restored = np_weighted_merge(np_restored, img_color_rgb, -weight)

    return np_restored


def np_adjust_chroma2(np_color_rgb: np.ndarray, np_gray_rgb: np.ndarray, hue_range: str = 'none',
                      return_mask: bool = False) -> np.ndarray:
    if hue_range == 'none' or hue_range == '':
        return np_color_rgb

    hsv_color = cv2.cvtColor(np_color_rgb, cv2.COLOR_RGB2HSV)
    hsv_s = hsv_color[:, :, 0]

    cond = _build_hue_conditions(hsv_s, hue_range)

    hsv_mask = np.where(cond, 255, 0)  # white only gray pixels

    mask_rgb = np_color_rgb.copy()

    for i in range(3):
        mask_rgb[:, :, i] = hsv_mask

    if return_mask:
        return Image.fromarray(mask_rgb, 'RGB').convert('RGB')

    np_restored = np_image_mask_merge(np_color_rgb, np_gray_rgb, mask_rgb)

    return np_restored


def _parse_hue_adjust(hue_adjust: str = 'none') -> ():
    p = hue_adjust.split("|")

    hue_range = ""
    sat = 1.0
    hue = 0
    weight = 0

    num = len(p)
    if num < 1 or num > 2:
        return None

    hue_range = p[0]

    if num == 1:
        return hue_range, sat, hue, weight

    sw = p[1].split(",")

    if (sw[0])[0] in ('-', '+'):
        hue = int(sw[0])
    else:
        sat = float(sw[0])

    if sat > 10:  # fix wrong input
        hue = int(sat)
        sat = 1.0

    weight = float(sw[1])

    return hue_range, sat, hue, weight


def _build_hue_conditions(hsv_s: np.ndarray = None, hue_range: str = None) -> np.ndarray:
    h_range = hue_range.split(",")
    h_len = len(h_range)

    hue_min, hue_max = _parse_hue_range(h_range[0])
    # For the 8-bit images, H is converted to H/2 to fit to the [0,255] range.
    c1 = hsv_s > hue_min * 0.5
    c2 = hsv_s < hue_max * 0.5
    cond = (c1 & c2)

    for i in range(1, h_len):
        hue_min, hue_max = _parse_hue_range(h_range[i])
        c1 = hsv_s > hue_min * 0.5
        c2 = hsv_s < hue_max * 0.5
        cond |= (c1 & c2)

    return cond


def _parse_hue_range(hue_range: str = None) -> ():
    # For color increments, each block in a given "hue_range" represents a Hue change of 30.
    match hue_range:
        case "red":
            rng = (0, 30)
        case "orange":
            rng = (30, 60)
        case "yellow":
            rng = (60, 90)
        case "yellow-green":
            rng = (90, 120)
        case "green":
            rng = (120, 150)
        case "blue-green":
            rng = (150, 180)
        case "cyan":
            rng = (180, 210)
        case "blue":
            rng = (210, 240)
        case "blue-violet":
            rng = (240, 270)
        case "violet":
            rng = (270, 300)
        case "red-violet":
            rng = (300, 330)
        case "rose":
            rng = (330, 360)
        case _:
            p = hue_range.split(":")
            if len(p) == 2 and p[0].isnumeric() and p[1].isnumeric():
                rng = (float(p[0]), float(p[1]))
            else:
                raise vs.Error("HybridAVC: unknown hue name: " + hue_range)

    return rng


def get_color_tune(hue_name: str = None) -> str:
    # For color increments, each block in a given "hue_range" represents a Hue change of 30.
    match hue_name:
        case "magenta":
            rng = "270:300"
        case "magenta/violet":
            rng = "270:330"
        case "violet":
            rng = "300:330"
        case "violet/red":
            rng = "300:360"
        case "blue/magenta":
            rng = "240:300"
        case "yellow":
            rng = "60:90"
        case "yellow/orange":
            rng = "30:90"
        case "yellow/green":
            rng = "60:120"
        case _:
            raise vs.Error("HybridAVC: unknown color tune: " + hue_name)

    return rng
