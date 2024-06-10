"""
------------------------------------------------------------------------------- 
Author: Dan64
Date: 2024-04-08
version: 
LastEditors: Dan64
LastEditTime: 2024-05-26
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
module containing the functions used to combine deoldify() and ddcolor().
"""
import vapoursynth as vs
import math
import os
import numpy as np
import cv2
from PIL import Image
from functools import partial

from .vsfilters import *
from .imfilters import *
from .vsutils import *
from .vsmodels import *

"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
function to build the refrence image used for the inference by DeepEx
"""

def vs_ext_reference_clip(clip: vs.VideoNode, sc_framedir: str = None) -> vs.VideoNode:

    ref_images = get_ref_images(sc_framedir)
    ref_images.sort()

    def set_clip_frame(n, f, img_list: list = None, f_size: Tuple[int, int] = None):

        is_scenechange = (n == 0) or (f.props['_SceneChangePrev'] == 1)

        if not is_scenechange:
            return f.copy()

        img_name = f"ref_{n:06d}"
        ext_ref_img = [f for f in img_list if img_name in f]
        if len(ext_ref_img) > 0:
            img_path = ext_ref_img[0]
            if os.path.isfile(img_path):
                try:
                    ref_img = Image.open(img_path).convert('RGB')
                    if (ref_img.size != f_size):
                        ref_img = ref_img.resize(f_size, Image.Resampling.LANCZOS)
                        #vs.core.log_message(2, "Resized reference frame: " + img_path + " size= " + str(f_size))
                except Exception as error:
                    vs.core.log_message(2, "Error reading reference frame: " + img_path + " -> " + str(error))
                    f.copy()
        else:
            return f.copy()

        return image_to_frame(ref_img, f.copy())

    clip_ref = clip.std.ModifyFrame(clips=[clip], selector=partial(set_clip_frame, img_list=ref_images, f_size=(clip.width, clip.height)))

    return clip_ref


def vs_reference_clip(clip: vs.VideoNode, method: int = 1, mweight: float = 0.4, render_factor=0,
        deoldify_p: list = [0, 24, 1.0, 0.0], ddcolor_p: list = [1, 24, 1.0, 0.0, True], ddtweak: bool = False,
        ddtweak_p: list = [0.0, 1.0, 2.5, True, 0.3, 0.6, 1.5, 0.5, "none"], scenechange: bool = True,
        device_index: int = 0, package_dir: str = None) -> vs.VideoNode:

    merge_weight = mweight

    # unpack deoldify_params
    deoldify_model = deoldify_p[0]
    if render_factor==0:
        deoldify_rf = deoldify_p[1]
    else:
        deoldify_rf = render_factor
    deoldify_sat = deoldify_p[2]
    deoldify_hue = deoldify_p[3]

    # unpack deoldify_params
    ddcolor_model = ddcolor_p[0]
    if render_factor==0:
        ddcolor_rf = ddcolor_p[1]
    else:
        ddcolor_rf = render_factor
    ddcolor_sat = ddcolor_p[2]
    ddcolor_hue = ddcolor_p[3]
    ddcolor_enable_fp16 = ddcolor_p[4]

    if ddcolor_rf != 0 and ddcolor_rf not in range(10, 65):
        raise vs.Error("HybridAVC: ddcolor render_factor must be between: 10-64")

    if ddcolor_rf == 0:
        ddcolor_rf = min(max(math.trunc(0.4 * clip.width / 16), 16), 48)

    clipb_weight = merge_weight

    clipa = vs_sc_deoldify(clip, method=method, model=deoldify_model, render_factor=deoldify_rf, scenechange=scenechange, package_dir=package_dir)

    clipb = vs_sc_ddcolor(clip, method=method, model=ddcolor_model, render_factor=ddcolor_rf, tweaks_enabled=ddtweak,
                       tweaks=ddtweak_p, enable_fp16=ddcolor_enable_fp16, scenechange=scenechange, device_index=device_index)

    return clipb

    clip_colored = vs_sc_combine_models(clipa, clipb, method=method, sat=[deoldify_sat, ddcolor_sat], scenechange=scenechange,
                                     hue=[deoldify_hue, ddcolor_hue], clipb_weight=merge_weight)

    return clip_colored

"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
main function used to combine the colored images with deoldify() and ddcolor()
"""
def vs_sc_combine_models(clipa: vs.VideoNode = None, clipb: vs.VideoNode = None, method: int = 0, sat: list = [1,1], hue: list = [0,0],
                      clipb_weight: float = 0.6, scenechange: bool = True) -> vs.VideoNode:

    #vs.core.log_message(2, "combine_models: method=" + str(method) + ", clipa = " + str(clipa) + ", clipb = " + str(clipb))

    if clipa is not None:
        clipa = vs_sc_tweak(clipa, hue=hue[0], sat=sat[0], scenechange=scenechange)
        if clipb is None: return clipa
    
    if clipb is not None:
        clipb = vs_sc_tweak(clipb, hue=hue[1], sat=sat[1], scenechange=scenechange)
        if clipa is None: return clipb

    if method == 2:
        return SCSimpleMerge(clipa, clipb, clipb_weight, scenechange)
    else:
        raise vs.Error("HybridAVC: only dd_method=(0,5) is supported")


def vs_combine_models(clip_a: vs.VideoNode = None, clip_b: vs.VideoNode = None, method: int = 0, sat: list = [1, 1],
                      hue: list = [0, 0], clipb_weight: float = 0.5, CMC_p: float = 0.2, LMM_p: list = [0.3, 0.6, 1.0],
                      ALM_p: list = [0.3, 0.6, 1.0], invert_clips: bool = False) -> vs.VideoNode:
    # vs.core.log_message(2, "combine_models: method=" + str(method) + ", clipa = " + str(clipa) + ", clipb = " + str(clipb))

    # unpack combine_params
    chroma_threshold = CMC_p
    luma_mask_limit = LMM_p[0]
    luma_white_limit = LMM_p[1]
    luma_mask_sat = LMM_p[2]
    luma_threshold = ALM_p[0]
    alpha = ALM_p[1]
    min_weight = ALM_p[2]

    if invert_clips:
        clipa = clip_b
        clipb = clip_a
    else:
        clipa = clip_a
        clipb = clip_b

    if clipa is not None:
        clipa = vs_tweak(clipa, hue=hue[0], sat=sat[0])
        if clipb is None: return clipa

    if clipb is not None:
        clipb = vs_tweak(clipb, hue=hue[1], sat=sat[1])
        if clipa is None: return clipb

    if method == 2:
        return SimpleMerge(clipa, clipb, clipb_weight)
    if method == 3:
        return ConstrainedChromaMerge(clipa, clipb, clipb_weight, chroma_threshold)
    if method == 4:
        return LumaMaskedMerge(clipa, clipb, luma_mask_limit, luma_white_limit, luma_mask_sat, clipb_weight)
    if method == 5:
        return AdaptiveLumaMerge(clipa, clipb, luma_threshold, alpha, clipb_weight, min_weight)
    else:
        raise vs.Error("HybridAVC: only dd_method=(0,5) is supported")

"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
the images are combined using a weighted merge, where the parameter clipb_weight
represent the weight assigned to the colors provided by ddcolor() 
"""

def SCSimpleMerge(clipa: vs.VideoNode = None, clipb: vs.VideoNode = None, clipb_weight: float = 0.5, scenechange: bool = True) -> vs.VideoNode:

    def merge_frame(n, f, weight: float = 0.5, scenechange: bool = True):

        if scenechange:
            is_scenechange = (n == 0) or (f[0].props['_SceneChangePrev'] == 1 and f[0].props['_SceneChangeNext'] == 0)
            if not is_scenechange:
                return f[0].copy()

        img1 = frame_to_image(f[0])
        img2 = frame_to_image(f[1]) 
        img_m = image_weighted_merge(img1, img2, weight)        
        return image_to_frame(img_m, f[0].copy())                

    clipm = clipa.std.ModifyFrame(clips=[clipa, clipb], selector=partial(merge_frame, weight=clipb_weight, scenechange=scenechange))

    return clipm

def SimpleMerge(clipa: vs.VideoNode = None, clipb: vs.VideoNode = None, clipb_weight: float = 0.5) -> vs.VideoNode:

    def merge_frame(n, f, weight: float = 0.5):
        img1 = frame_to_image(f[0])
        img2 = frame_to_image(f[1])
        img_m = image_weighted_merge(img1, img2, weight)
        return image_to_frame(img_m, f[0].copy())
    clipm = clipa.std.ModifyFrame(clips=[clipa, clipb], selector=partial(merge_frame, weight=clipb_weight))
    return clipm


"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
the clips are combined using a mask merge, the pixels of clipb with luma < luma_mask_limit
will be filled with the pixels of clipa, if the parameter clipm_weight > 0
the masked image will be merged with clipa 
"""

def LumaMaskedMerge(clipa: vs.VideoNode = None, clipb: vs.VideoNode = None, luma_mask_limit: float = 0.4,
                    luma_white_limit: float = 0.7, luma_mask_sat=1.0, clipm_weight: float = 0.5) -> vs.VideoNode:
    if luma_mask_sat < 1:
        # vs.core.log_message(2, "LumaMaskedMerge: mask_sat = " + str(luma_mask_sat))
        clipc = vs_tweak(clipa, sat=luma_mask_sat)
    else:
        clipc = clipa

    def merge_frame(n, f, weight: float = 0.5, luma_limit: float = 0.4, white_limit: float = 0.7):
        img1 = frame_to_image(f[0])
        img2 = frame_to_image(f[1])
        img3 = frame_to_image(f[2])
        if luma_limit == white_limit:
            # vs.core.log_message(2, "frame[" + str(n) + "]: luma_limit = " + str(luma_limit))
            img_masked = image_luma_merge(img3, img2, luma_limit)
        else:
            img_masked = w_image_luma_merge(img3, img2, luma_limit, white_limit)
        if clipm_weight < 1:
            img_m = image_weighted_merge(img1, img_masked, weight)
        else:
            img_m = img_masked
        return image_to_frame(img_m, f[0].copy())

    clipm = clipa.std.ModifyFrame(clips=[clipa, clipb, clipc],
                                  selector=partial(merge_frame, weight=clipm_weight, luma_limit=luma_mask_limit,
                                                   white_limit=luma_white_limit))
    return clipm


"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
given the ddcolor() perfomance is quite bad on dark scenes, the images are 
combinaed by decreasing the weight assigned to ddcolor() when the luma is 
below a given threshold given by: luma_threshold. 
For example with: luma_threshold = 0.6 and alpha = 1, the weight assigned to 
ddcolor() will start to decrease linearly when the luma < 60% till "min_weight".
For alpha=2, begins to decrease quadratically.      
"""

def AdaptiveLumaMerge(clipa: vs.VideoNode = None, clipb: vs.VideoNode = None, luma_threshold: float = 0.6,
                      alpha: float = 1.0, clipb_weight: float = 0.5, min_weight: float = 0.15) -> vs.VideoNode:
    def merge_frame(n, f, luma_limit: float = 0.6, min_w: float = 0.15, alpha: float = 1.0, weight: float = 0.5):
        img1 = frame_to_image(f[0])
        img2 = frame_to_image(f[1])
        luma = get_image_luma(img2)
        if luma < luma_limit:
            bright_scale = pow(luma / luma_limit, alpha)
            w = max(weight * bright_scale, min_w)
        else:
            w = weight
        # vs.core.log_message(2, "Luma(" + str(n) + ") = " + str(luma) + ", weight = " + str(w))
        img_m = Image.blend(img1, img2, w)
        return image_to_frame(img_m, f[0].copy())

    clipm = clipa.std.ModifyFrame(clips=[clipa, clipb],
                                  selector=partial(merge_frame, luma_limit=luma_threshold, min_w=min_weight,
                                                   alpha=alpha, weight=clipb_weight))
    return clipm

"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
given that the colors provided by deoldify() are more conservative and stable 
than the colors obtained with ddcolor() images are combined by assigning
a limit to the amount of difference in chroma values between deoldify() and
ddcolor() this limit is defined by the parameter threshold. The limit is applied
to the image converted to "YUV". For example when threshold=0.1, the chroma
values "U","V" of ddcolor() image will be constrained to have an absolute
percentage difference respect to "U","V" provided by deoldify() not higher than 10%    
"""

def ConstrainedChromaMerge(clipa: vs.VideoNode = None, clipb: vs.VideoNode = None, clipb_weight: float = 0.5,
                           chroma_threshold: float = 0.2) -> vs.VideoNode:
    def merge_frame(n, f, level: float = 0.2, weight: float = 0.5):
        img1 = frame_to_image(f[0])
        img2 = frame_to_image(f[1])
        img_m = chroma_stabilizer(img1, img2, level, weight)
        return image_to_frame(img_m, f[0].copy())

    clipm = clipa.std.ModifyFrame(clips=[clipa, clipb],
                                  selector=partial(merge_frame, level=chroma_threshold, weight=clipb_weight))
    return clipm
