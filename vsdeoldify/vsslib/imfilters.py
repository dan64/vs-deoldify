"""
------------------------------------------------------------------------------- 
Author: Dan64
Date: 2024-04-08
version: 
LastEditors: Dan64
LastEditTime: 2025-10-04
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
Library of filter functions working on images.
"""
import numpy as np
import cv2
from PIL import Image, ImageMath, ImageEnhance
from vsdeoldify.vsslib.nputils import np_rgb_to_gray, np_image_mask_merge, w_np_rgb_to_gray, w_np_image_mask_merge
from vsdeoldify.vsslib.nputils import  array_clip, np_image_gamma_contrast, np_hue_add
from vsdeoldify.vsslib.restcolor import np_adjust_chroma2, np_image_chroma_tweak

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

    if dark_luma >= white_luma:
        return img_dark

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

    if weight == 0.0:
        return img1

    if weight == 1.0:
        return img2

    # img_m = img1 * (1.0 - w) + img2 * w
    img_m = Image.blend(img1, img2, weight)  # faster

    return img_m
"""
    img1_np = np.asarray(img1)
    img2_np = np.asarray(img2)

    img_new = np_image_weighted_merge(img1_np, img2_np, weight)

    return Image.fromarray(img_new)
"""

def np_image_weighted_merge(img1_np: np.ndarray, img2_np: np.ndarray, weight: float = 0.5) -> np.ndarray:
    img_new = np.copy(img1_np)

    img_m = np.multiply(img1_np, 1 - weight) + np.multiply(img2_np, weight)

    img_m = np.uint8(np.clip(img_m, 0, 255))
    # img_m = img_m.clip(0, 255).astype(int)

    img_new[:, :, 0] = img_m[:, :, 0]
    img_new[:, :, 1] = img_m[:, :, 1]
    img_new[:, :, 2] = img_m[:, :, 2]

    return img_new


"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
Function to limit the chroma of "img_new" to have an absolute percentage 
difference respect to "U","V" provided by "img_stable" not higher than "alpha"  
"""


def chroma_stabilizer(img_stable: Image, img_new: Image, alpha: float = 0.15, weight: float = 1.0) -> Image:
    """
    Function to limit the chroma of "img_new" to have an absolute percentage
    difference respect to "U","V" provided by "img_stable" not higher than "alpha"
    """
    img1_np = np.asarray(img_stable)
    yuv1 = cv2.cvtColor(img1_np, cv2.COLOR_RGB2YUV)
    y1 = yuv1[:, :, 0]
    u1 = yuv1[:, :, 1]
    v1 = yuv1[:, :, 2]

    u1_up = np.multiply(u1, 1 + alpha).clip(0, 255).astype(np.uint8)
    v1_up = np.multiply(v1, 1 + alpha).clip(0, 255).astype(np.uint8)

    u1_dn = np.multiply(u1, 1 - alpha).clip(0, 255).astype(np.uint8)
    v1_dn = np.multiply(v1, 1 - alpha).clip(0, 255).astype(np.uint8)

    img2_np = np.asarray(img_new)
    yuv2 = cv2.cvtColor(img2_np, cv2.COLOR_RGB2YUV)

    v2_new = np.copy(yuv2)

    u2 = yuv2[:, :, 1]
    v2 = yuv2[:, :, 2]

    u2_m = array_clip(u2, u1_dn, u1_up, np.uint8)
    v2_m = array_clip(v2, v1_dn, v1_up, np.uint8)

    v2_new[:, :, 0] = y1
    v2_new[:, :, 1] = u2_m
    v2_new[:, :, 2] = v2_m

    # Convert back to RGB
    rgb_out = cv2.cvtColor(v2_new, cv2.COLOR_YUV2RGB)
    img_out = Image.fromarray(rgb_out)

    # Optional blending
    if weight < 1.0:
        return Image.blend(img_stable, img_out, weight)
    else:
        return img_out

def chroma_stabilizer_adaptive(
    img_stable: Image.Image,
    img_new: Image.Image,
    base_tol: int = 18,
    max_extra: int = 22,
    weight: float = 1.0
) -> Image.Image:
    """
        In OpenCV’s 8-bit YUV (aka YCrCb):

        Y (luma): 0–255 (0 = black, 255 = white)
        U (Cb): 0–255, but neutral gray is at 128
        V (Cr): 0–255, neutral gray is at 128
    So:
        (U, V) = (128, 128) → achromatic (gray)
        (U, V) = (80, 160) → reddish
        (U, V) = (160, 80) → greenish

    The meaningful chroma signal is actually (U − 128, V − 128), which ranges from −128 to +127.
    """
    # Convert to YUV
    img1_np = np.asarray(img_stable).astype(np.uint8)

    yuv1 = cv2.cvtColor(img1_np, cv2.COLOR_RGB2YUV)

    y1 = yuv1[:, :, 0].astype(np.float32)
    u1 = yuv1[:, :, 1].astype(np.int16) - 128
    v1 = yuv1[:, :, 2].astype(np.int16) - 128

    img2_np = np.asarray(img_new).astype(np.uint8)
    yuv2 = cv2.cvtColor(img2_np, cv2.COLOR_RGB2YUV)
    u2 = yuv2[:, :, 1].astype(np.int16) - 128
    v2 = yuv2[:, :, 2].astype(np.int16) - 128

    # Compute texture strength via Laplacian
    lap = cv2.Laplacian(y1, cv2.CV_32F)
    texture = np.abs(lap) / 255.0
    texture = np.clip(texture, 0.0, 1.0)

    # Adaptive tolerance per pixel
    chroma_tol = base_tol + max_extra * texture

    # Compute bounds
    u_low = np.clip(u1 - chroma_tol, -128, 127)
    u_high = np.clip(u1 + chroma_tol, -128, 127)
    v_low = np.clip(v1 - chroma_tol, -128, 127)
    v_high = np.clip(v1 + chroma_tol, -128, 127)

    # Constrain new chroma
    u2_m = np.clip(u2, u_low, u_high)
    v2_m = np.clip(v2, v_low, v_high)

    # Reconstruct YUV
    yuv_out = np.stack([
        yuv1[:, :, 0],
        (u2_m + 128).astype(np.uint8),
        (v2_m + 128).astype(np.uint8)
    ], axis=2)

    # Convert back to RGB
    rgb_out = cv2.cvtColor(yuv_out, cv2.COLOR_YUV2RGB)
    img_out = Image.fromarray(rgb_out)

    # Optional blending
    if weight < 1.0:
        return Image.blend(img_stable, img_out, weight)
    else:
        return img_out

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
    g_m = ImageMath.eval("convert(max(min(a, c), b), 'L')", a=g1_up, b=g1_dn, c=g2)
    b_m = ImageMath.eval("convert(max(min(a, c), b), 'L')", a=b1_up, b=b1_dn, c=b2)

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
        y_new = np.add(y, i_alpha).clip(i_min, i_max).astype(np.uint8)
    else:
        y_new = y

    if gamma != 1 and luma < gamma_luma_min:
        if gamma_alpha != 0:
            g_new = max(gamma * pow(luma / gamma_luma_min, gamma_alpha), gamma_min)
        else:
            g_new = gamma
        y_new = np.power(y_new / 255, 1 / g_new)
        y_new = np.multiply(y_new, 255).clip(i_min, i_max).astype(np.uint8)

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


def image_gamma_contrast(img: Image, gamma: float = 1.0, cont: float = 1.0):
    if cont == 1 and gamma == 1:
        return img

    img_np = np.asarray(img)

    np_img_rgb = np_image_gamma_contrast(img_np, gamma, cont)

    return Image.fromarray(np_img_rgb)


def image_contrast(img: Image, cont: float = 1.0):
    if cont == 1:
        return img

    return image_gamma_contrast(img, cont)


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

    y_cont = array_clip(y_cont, 0, 1, np.float32) * 255

    yuv_new = np.copy(yuv)

    yuv_new[:, :, 0] = y_cont.clip(0, 255).astype(np.uint8)

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

"""
def image_tweak(img: Image, sat: float = 1, cont: float = 1.0, bright: float = 0, hue: float = 0, gamma: float = 1.0,
                hue_range: str = 'none') -> Image:
    if sat == 1 and bright == 0 and hue == 0 and gamma == 1 and cont == 1:
        return img  # non changes

    img_np = np.asarray(img)

    img_rgb = np_image_tweak(img_np, sat, cont, bright, hue, gamma, hue_range)

    return Image.fromarray(img_rgb, 'RGB').convert('RGB')
"""

def image_tweak(img: Image, sat: float = 1, cont: float = 1.0, bright: float = 0, hue: float = 0, gamma: float = 1.0,
                hue_range: str = 'none') -> Image.Image:
    """
    Adjust brightness, contrast, saturation, hue, and gamma of a PIL RGB image.

    Args:
        img: PIL Image (must be RGB)
        bright: 0.0 = original (-128 = 50% darker, +128 = 50% brighter)
        cont:   1.0 = original
        sat: 1.0 = original (0 = grayscale)
        hue:   degrees to rotate hue (-180 to 180)
        gamma:  1.0 = original (<1 = brighter, >1 = darker)
        hue_range: if not 'none' the adjustments will be applied only on the hue range specified

    Returns:
        Adjusted PIL Image (RGB)
    """
    img_np = np.asarray(img)

    # Step 1: Apply gamma correction (pixel-wise, before color adjustments)
    if gamma != 1.0:
        img = _apply_gamma(img, gamma)

    # Step 2: Apply hue shift (requires HSV conversion)
    if hue != 0.0:
        img = _apply_hue_shift(img, hue)

    # Step 3: Apply PIL enhancements (order matters: brightness → contrast → saturation)
    if bright != 0.0:
        brightness = 1 + bright/255
        img = ImageEnhance.Brightness(img).enhance(brightness)
    if cont != 1.0:
        img = ImageEnhance.Contrast(img).enhance(cont)
    if sat != 1.0:
        img = ImageEnhance.Color(img).enhance(sat)

    if hue_range == 'none' or hue_range == '':
        return img

    img_new_np = np_adjust_chroma2(img_np, np.asarray(img), hue_range)

    return Image.fromarray(img_new_np, mode='RGB')


def _apply_gamma(img: Image.Image, gamma: float) -> Image.Image:
    """Apply gamma correction using lookup table (fast)."""
    # Build gamma lookup table
    inv_gamma = 1.0 / gamma
    table = np.array([
        ((i / 255.0) ** inv_gamma) * 255
        for i in range(256)
    ]).astype("uint8")

    # Apply to all channels
    return img.point(table * 3)


def _apply_hue_shift(img: Image.Image, hue_deg: float) -> Image.Image:
    """Shift hue by hue_deg degrees using HSV conversion."""
    # Convert to HSV
    img_hsv = img.convert('HSV')
    h, s, v = img_hsv.split()

    # Convert hue to numpy array
    h_np = np.array(h, dtype=np.int16)  # use int16 to avoid overflow

    # Shift hue (HSV hue is 0-255, not 0-360!)
    # PIL's HSV: H ∈ [0, 255] ≈ [0°, 360°]
    hue_offset = int((hue_deg / 360.0) * 255)
    h_np = (h_np + hue_offset) % 256

    # Convert back to uint8 and image
    h_new = Image.fromarray(h_np.astype('uint8'), mode='L')
    img_hsv_shifted = Image.merge('HSV', (h_new, s, v))

    return img_hsv_shifted.convert('RGB')

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


def get_image_luma(img: Image, maxrange: int = 255) -> float:
    img_np = np.asarray(img)
    yuv = cv2.cvtColor(img_np, cv2.COLOR_RGB2YUV)
    luma = np.mean(yuv[:, :, 0])
    return round(luma / maxrange, 6)

"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
image blend based on luma 
"""

def image_luma_blend(img: Image, img_new: Image, f_luma: float = 0.5, luma_limit: float = 0.4,
                     alpha: float = 0.90, min_w: float = 0.15, decay: float = 4.0) -> Image:

    # Luma merge
    if f_luma < luma_limit:
        bright_scale = min(max(pow(f_luma / luma_limit, decay), 0), 1)
        w = round(max(alpha * bright_scale, min_w), 6)
        # img_m = img * (1.0 - w) + img_new * w
        img_m = Image.blend(img, img_new, w)
    else:
        img_m = img_new

    return img_m


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

    u2_m = array_clip(u2, u1_dn, u1_up)
    v2_m = array_clip(v2, v1_dn, v1_up)

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
