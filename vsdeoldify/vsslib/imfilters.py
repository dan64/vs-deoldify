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
Library of filter functions working on images.
"""
import math
import numpy as np
import cv2
from PIL import Image, ImageMath

from .nputils import *
from .restcolor import np_adjust_chroma2
from .restcolor import np_image_chroma_tweak

"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
convert an image to gray or B&W if threshold > 0 
"""


def rgb_to_gray(img: Image, threshold: float = 0) -> Image:
    gray_np = np_rgb_to_gray(np.array(img), threshold)

    return Image.fromarray(gray_np, 'RGB')


"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
merge image1 with image2 using the image mask (white->img2, black->img1) 
"""


def image_mask_merge(img1: Image, img2: Image, mask: Image) -> Image:
    img1_np = np.array(img1)
    img2_np = np.array(img2)
    mask_np = np.array(mask)

    img_np = np_image_mask_merge(img1_np, img2_np, mask_np)

    return Image.fromarray(img_np, 'RGB')


"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
merge image1 with image2 using the image mask (mask_white->img_white, mask_black->img_dark) 
"""


def image_luma_merge(img_dark: Image, img_white: Image, luma: float = 0, return_mask: bool = False) -> Image:
    img1_np = np.array(img_dark)
    img2_np = np.array(img_white)
    # the mask is built using the second image
    mask_np = np_rgb_to_gray(img2_np, luma)

    if return_mask:
        return Image.fromarray(mask_np, 'RGB')

    img_np = np_image_mask_merge(img1_np, img2_np, mask_np)

    return Image.fromarray(img_np, 'RGB')


def w_image_luma_merge(img_dark: Image, img_white: Image, dark_luma: float = 0.3, white_luma=0.9,
                       return_mask: bool = False) -> Image:
    img1_np = np.array(img_dark)
    img2_np = np.array(img_white)
    # the mask is built using the second image
    mask_w_np = w_np_rgb_to_gray(img2_np, dark_luma, white_luma)

    if return_mask:
        mask_np = np.multiply(mask_w_np, 255).clip(0, 255).astype(int)
        img_mask = img1_np.copy()
        for i in range(3):
            img_mask[:, :, i] = mask_np[:, :, i]
        return Image.fromarray(img_mask, 'RGB')

    img_np = w_np_image_mask_merge(img1_np, img2_np, mask_w_np)

    return Image.fromarray(img_np, 'RGB')


"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
numpy implementation of image merge on 3 planes, faster than vs.core.std.Merge()
"""


def image_weighted_merge(img1: Image, img2: Image, weight: float = 0.5) -> Image:
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


"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
Function to limit the chroma of "img_new" to have an absolute percentage 
difference respect to "U","V" provided by "img_stable" not higher than "alpha"  
"""


def chroma_stabilizer(img_stable: Image, img_new: Image, alpha: float = 0.2, weight: float = 1.0) -> Image:
    img1_np = np.asarray(img_stable)
    yuv1 = cv2.cvtColor(img1_np, cv2.COLOR_RGB2YUV)
    y1 = yuv1[:, :, 0]
    u1 = yuv1[:, :, 1]
    v1 = yuv1[:, :, 2]

    u1_up = np.multiply(u1, 1 + alpha).clip(0, 255).astype(int)
    v1_up = np.multiply(v1, 1 + alpha).clip(0, 255).astype(int)

    u1_dn = np.multiply(u1, 1 - alpha).clip(0, 255).astype(int)
    v1_dn = np.multiply(v1, 1 - alpha).clip(0, 255).astype(int)

    img2_np = np.asarray(img_new)
    yuv2 = cv2.cvtColor(img2_np, cv2.COLOR_RGB2YUV)

    v2_new = np.copy(yuv2)

    u2 = yuv2[:, :, 1]
    v2 = yuv2[:, :, 2]

    u2_m = array_max_min(u2, u1_up, u1_dn)
    v2_m = array_max_min(v2, v1_up, v1_dn)

    v2_new[:, :, 0] = y1
    v2_new[:, :, 1] = u2_m
    v2_new[:, :, 2] = v2_m

    if weight < 1.0:
        v2_m = np.multiply(yuv1, 1 - weight).clip(0, 255).astype(int) + np.multiply(v2_new, weight).clip(0, 255).astype(
            int)
        v2_new[:, :, 0] = v2_m[:, :, 0]
        v2_new[:, :, 1] = v2_m[:, :, 1]
        v2_new[:, :, 2] = v2_m[:, :, 2]

    v2_rgb = cv2.cvtColor(v2_new, cv2.COLOR_YUV2RGB)

    return Image.fromarray(v2_rgb)


"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
Implementation of function chroma_stabilizer() with fixed threshold
of 20% using the Pillow library (slower than chroma_stabilizer)   
"""


def chroma_smoother(img_prv: Image, img: Image) -> Image:
    r2, g2, b2 = img.split()

    img1_up = Image.eval(img_prv, (lambda x: min(x * (1 + 0.20), 255)))
    img1_dn = Image.eval(img_prv, (lambda x: max(x * (1 - 0.20), 0)))

    r1_up, g1_up, b1_up = img1_up.split()
    r1_dn, g1_dn, b1_dn = img1_dn.split()

    r_m = ImageMath.eval("convert(max(min(a, c), b), 'L')", a=r1_up, b=r1_dn, c=r2)
    g_m = ImageMath.eval("convert(max(min(a, c), b), 'L')", a=g1_up, b=g1_dn, c=r2)
    b_m = ImageMath.eval("convert(max(min(a, c), b), 'L')", a=b1_up, b=b1_dn, c=r2)

    img_m = Image.merge('RGB', (r_m, g_m, b_m))

    img_final = chroma_post_process(img_m, img)

    return img_final


"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
Function to copy the chroma parametrs "U", "V", of "img_m" in "orig" 
"""


def chroma_post_process(img_m: Image, orig: Image) -> Image:
    img_np = np.asarray(img_m)
    orig_np = np.asarray(orig)
    img_yuv = cv2.cvtColor(img_np, cv2.COLOR_RGB2YUV)
    # copy the chroma parametrs "U", "V", of "img_m" in "orig" 
    orig_yuv = cv2.cvtColor(orig_np, cv2.COLOR_RGB2YUV)
    orig_copy = np.copy(orig_yuv)
    orig_copy[:, :, 1:3] = img_yuv[:, :, 1:3]
    img_np_new = cv2.cvtColor(orig_copy, cv2.COLOR_YUV2RGB)
    return Image.fromarray(img_np_new)


"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
This function force the average luma of an image to don't be below the value
defined by the parameter "luma_min". The function allow to modify the gamma
of image if the average luma is below the parameter "gamma_luma_min"  
"""


def luma_adjusted_levels(img: Image, luma_min: float = 0, gamma: float = 1.0, gamma_luma_min: float = 0,
                         gamma_alpha: float = 0, gamma_min: float = 0.2, i_min: int = 0, i_max: int = 255) -> Image:
    img_np = np.asarray(img)

    yuv = cv2.cvtColor(img_np, cv2.COLOR_RGB2YUV)

    y = yuv[:, :, 0]
    u = yuv[:, :, 1]
    v = yuv[:, :, 2]

    luma = np.mean(yuv[:, :, 0]) / 255

    if luma < luma_min:
        i_alpha = int(255 * (luma_min - luma))
    else:
        i_alpha = 0

    yuv_new = np.copy(yuv).clip(i_min, i_max)

    if i_alpha > 1:
        y_new = np.add(y, i_alpha).clip(i_min, i_max).astype(int)
    else:
        y_new = y

    if gamma != 1 and luma < gamma_luma_min:
        if gamma_alpha != 0:
            g_new = max(gamma * pow(luma / gamma_luma_min, gamma_alpha), gamma_min)
        else:
            g_new = gamma
        y_new = np.power(y_new / 255, 1 / g_new)
        y_new = np.multiply(y_new, 255).clip(i_min, i_max).astype(int)

    yuv_new[:, :, 0] = y_new
    yuv_new[:, :, 1] = u
    yuv_new[:, :, 2] = v

    rgb_np = cv2.cvtColor(yuv_new, cv2.COLOR_YUV2RGB)
    return Image.fromarray(rgb_np)


"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
adjust the contrast of an image, color-space: YUV 
"""


def image_gamma_contrast(img: Image, gamma: float = 1.0, cont: float = 1.0, perc: float = 5):
    if cont == 1 and gamma == 1:
        return img

    img_np = np.asarray(img)

    np_img_rgb = np_image_gamma_contrast(img_np, gamma, cont)

    return Image.fromarray(np_img_rgb)


def image_contrast(img: Image, cont: float = 1.0, perc: float = 5):
    if cont == 1:
        return img

    return image_gamma_contrast(img, cont, perc)


"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
adjust the brightness of an image, color-space: YUV 
"""


def image_brightness(img: Image, bright: float = 0.0):
    if bright == 0:
        return img

    img_np = np.asarray(img)
    yuv = cv2.cvtColor(img_np, cv2.COLOR_RGB2YUV)

    y = yuv[:, :, 0]

    y_cont = y / 255 + bright

    y_cont = array_min_max(y_cont, 0, 1, np.float64) * 255

    yuv_new = np.copy(yuv)

    yuv_new[:, :, 0] = y_cont.clip(0, 255).astype(int)

    img_rgb = cv2.cvtColor(yuv_new, cv2.COLOR_YUV2RGB)

    return Image.fromarray(img_rgb)


"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
Simple function adjust hue and decrease saturation/brightness of an image.    
The ranges that OpenCV manage for HSV format are the following:
- Hue: range is [-180,+180], 
- Saturation: range is [0,10] 
- Value: range is [-1,10].
For the 8-bit images, H is converted to H/2 to fit to the [0,255] range. 
So the range of hue in the HSV color space of OpenCV is [-90,+90]
"""


def image_tweak(img: Image, sat: float = 1, cont: float = 1.0, bright: float = 0, hue: float = 0, gamma: float = 1.0,
                hue_range: str = 'none') -> Image:
    if sat == 1 and bright == 0 and hue == 0 and gamma == 1 and cont == 1:
        return img  # non changes

    img_np = np.asarray(img)

    img_rgb = np_image_tweak(img_np, sat, cont, bright, hue, gamma, hue_range)

    return Image.fromarray(img_rgb, 'RGB').convert('RGB')


def image_chroma_tweak(img: Image, sat: float = 1, bright: float = 0, hue: int = 0, hue_adjust: str = 'none') -> Image:
    if sat == 1 and bright == 0 and hue == 0 and hue_adjust == "none":
        return img  # non changes

    img_np = np.asarray(img)

    img_rgb = np_image_chroma_tweak(img_np, sat, bright, hue, hue_adjust)

    return Image.fromarray(img_rgb, 'RGB').convert('RGB')


def np_image_tweak(img_np: np.ndarray, sat: float = 1, cont: float = 1.0, bright: float = 0, hue: float = 0,
                   gamma: float = 1.0, hue_range: str = 'none') -> np.ndarray:
    if cont != 1 or gamma != 1:
        img_np = np_image_gamma_contrast(img_np, gamma, cont)

    if sat == 1 and bright == 0 and hue == 0 and hue_range == 'none':
        return img_np  # no other changes

    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)

    hsv[:, :, 0] = np_hue_add(hsv[:, :, 0], hue)
    hsv[:, :, 1] = hsv[:, :, 1] * min(max(sat, 0), 10)
    hsv[:, :, 2] = hsv[:, :, 2] * min(max(1 + bright, 0), 10)

    img_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    return np_adjust_chroma2(img_np, img_rgb, hue_range)


"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
get the value of average brightness of an image
"""


def get_image_brightness(img: Image) -> float:
    img_np = np.asarray(img)
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    brightness = np.mean(hsv[:, :, 2])
    return brightness / 255


"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
get the value of average luma of an image
"""


def get_image_luma(img: Image) -> float:
    img_np = np.asarray(img)
    yuv = cv2.cvtColor(img_np, cv2.COLOR_RGB2YUV)
    luma = np.mean(yuv[:, :, 0])
    return luma / 255


"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description: 
------------------------------------------------------------------------------- 
Temporal luma limiter: the function will limit the luma of "cur_img" to have an 
absolute percentage deviation respect to "prv_img" not higher than "alpha"  
"""


def _chroma_temporal_limiter(cur_img: Image, prv_img: Image, alpha: float = 0.05) -> Image:
    img1_np = np.asarray(prv_img)
    yuv1 = cv2.cvtColor(img1_np, cv2.COLOR_RGB2YUV)
    u1 = yuv1[:, :, 1]
    v1 = yuv1[:, :, 2]

    u1_up = np.multiply(u1, 1 + alpha)
    u1_dn = np.multiply(u1, 1 - alpha)

    v1_up = np.multiply(v1, 1 + alpha)
    v1_dn = np.multiply(v1, 1 - alpha)

    img2_np = np.asarray(cur_img)
    yuv2 = cv2.cvtColor(img2_np, cv2.COLOR_RGB2YUV)

    yuv_new = np.copy(yuv2)

    u2 = yuv2[:, :, 1]
    v2 = yuv2[:, :, 2]

    u2_m = array_max_min(u2, u1_up, u1_dn)
    v2_m = array_max_min(v2, v1_up, v1_dn)

    yuv_new[:, :, 1] = u2_m
    yuv_new[:, :, 2] = v2_m

    rgb_new = cv2.cvtColor(yuv_new, cv2.COLOR_YUV2RGB)

    return Image.fromarray(rgb_new)


"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
Temporal color stabilizer the UV chroma of current frame are averaged with the
values of previous "nframes"  
"""


def _color_temporal_stabilizer(img_f: list, weight_list: list = None) -> Image:
    nframes = len(weight_list)

    Nh = round((nframes - 1) / 2)

    img_new = np.copy(np.asarray(img_f[Nh]))

    yuv_new = cv2.cvtColor(img_new, cv2.COLOR_RGB2YUV)

    weight: float = weight_list[Nh] / 100.0

    yuv_m = np.multiply(yuv_new, weight)

    for i in range(0, Nh):
        yuv_i = cv2.cvtColor(np.asarray(img_f[i]), cv2.COLOR_RGB2YUV)
        weight: float = weight_list[i] / 100.0
        yuv_m += np.multiply(yuv_i, weight)
    for i in range(Nh + 1, nframes):
        yuv_i = cv2.cvtColor(np.asarray(img_f[i]), cv2.COLOR_RGB2YUV)
        weight: float = weight_list[i] / 100.0
        yuv_m += np.multiply(yuv_i, weight)

    yuv_new[:, :, 1] = yuv_m[:, :, 1]
    yuv_new[:, :, 2] = yuv_m[:, :, 2]

    return Image.fromarray(cv2.cvtColor(yuv_new, cv2.COLOR_YUV2RGB))
