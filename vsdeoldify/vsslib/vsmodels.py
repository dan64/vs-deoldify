"""
------------------------------------------------------------------------------- 
Author: Dan64
Date: 2024-04-08
version: 
LastEditors: Dan64
LastEditTime: 2024-05-08
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
module containing the main functions to colorize the frames with deoldify() and ddcolor().
"""
import vapoursynth as vs
import math
import numpy as np
import cv2
from PIL import Image
from functools import partial

from ..deoldify.visualize import *
from .vsutils import *
from .vsfilters import *

"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
wrapper to deoldify. 
"""
def vs_deoldify(clip: vs.VideoNode, method: int = 2, model: int = 0, render_factor: int = 24, tweaks_enabled: bool = False, tweaks: list = [0.0, 0.9, 0.7, False, 0.3, 0.3], package_dir: str = "") -> vs.VideoNode: 
    
    if method == 1:
        return None
    
    # unpack tweaks
    bright = tweaks[0]
    cont = tweaks[1]
    gamma = tweaks[2]
    luma_constrained_tweak=tweaks[3]
    luma_min = tweaks[4] 
    gamma_luma_min = tweaks[5]
    gamma_alpha = tweaks[6]
    gamma_min = tweaks[7]    
    if (len(tweaks) > 8):
        hue_adjust = tweaks[8]
    else:
        hue_adjust = 'none'
    
    if tweaks_enabled:     
        if luma_constrained_tweak:
            clipa = vs_tweak(clip, bright=bright, cont=cont) # contrast and bright are adjusted before the constrainded luma and gamma
            clipa = constrained_tweak(clipa, luma_min = luma_min, gamma=gamma, gamma_luma_min = gamma_luma_min, gamma_alpha = gamma_alpha, gamma_min=gamma_min)
        else:
            clipa = vs_tweak(clip, bright=bright, cont=cont, gamma=gamma)
    else:
        clipa = clip
        
    clipa_rgb =  _deoldify(clipa, model, render_factor, package_dir)    
    
    if tweaks_enabled and hue_adjust != 'none':
        clipa_rgb = vs_adjust_clip_hue(clipa_rgb, hue_adjust.lower())
    
    if tweaks_enabled:
        return vs_recover_clip_luma(clip, clipa_rgb)
    else:
        return clipa_rgb    

def _deoldify(clip: vs.VideoNode, model: int = 0, render_factor: int = 24, package_dir: str = "") -> vs.VideoNode: 
            
    match model:
        case 0:
            colorizer = get_image_colorizer(root_folder=Path(package_dir), artistic=False,isvideo=True) 
        case 1:
            colorizer = get_image_colorizer(root_folder=Path(package_dir), artistic=False,isvideo=False) 
        case 2:
            colorizer = get_image_colorizer(root_folder=Path(package_dir), artistic=True,isvideo=False) 
                    
    def deoldify_colorize(n: int, f: vs.VideoFrame, colorizer: ModelImageVisualizer = None, render_factor: int = 24) -> vs.VideoFrame:
        img_orig = frame_to_image(f)
        img_color = colorizer.get_transformed_image(img_orig, render_factor=render_factor, post_process=True)
        return image_to_frame(img_color, f.copy()) 
    
    return clip.std.ModifyFrame(clips=[clip], selector=partial(deoldify_colorize, colorizer=colorizer, render_factor=render_factor))         

"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
wrapper to function ddcolor() with tweak pre-process.
"""
def vs_ddcolor(clip: vs.VideoNode, method: int = 2, model: int = 0, render_factor: int = 24, tweaks_enabled: bool = False, tweaks: list = [0.0, 0.9, 0.7, False, 0.3, 0.3], dstrength: int = 0, enable_fp16: bool = True, device_index: int = 0, num_streams: int = 1) -> vs.VideoNode:
    
    if method == 0:
        return None
    else: 
        import vsddcolor
    
    # input size must a multiple of 32
    input_size = math.trunc(render_factor/2)*32
    
    try:
        d_clip = vs_degrain(clip, strength=dstrength, device_id=device_index)
    except Exception as error:
        vs.core.log_message(2, "ddeoldify: KNLMeansCL error -> " + str(error))   
        d_clip = clip    
    clip = d_clip 
    
    # unpack tweaks
    bright = tweaks[0]
    cont = tweaks[1]
    gamma = tweaks[2]
    luma_constrained_tweak=tweaks[3]
    luma_min = tweaks[4] 
    gamma_luma_min = tweaks[5]
    gamma_alpha = tweaks[6]
    gamma_min = tweaks[7]    
    if (len(tweaks) > 8):
        hue_adjust = tweaks[8]
    else:
        hue_adjust = 'none'
        
    if tweaks_enabled:     
        if luma_constrained_tweak:
            clipb = vs_tweak(clip, bright=bright, cont=cont) # contrast and bright are adjusted before the constrainded luma and gamma
            clipb = constrained_tweak(clipb, luma_min = luma_min, gamma=gamma, gamma_luma_min = gamma_luma_min, gamma_alpha = gamma_alpha, gamma_min=gamma_min)
        else:
            clipb = vs_tweak(clip, bright=bright, cont=cont, gamma=gamma)
    else:
        clipb = clip       
    # adjusting clip's color space to RGBH for vsDDColor
    if enable_fp16:
        clipb = vsddcolor.ddcolor(clipb.resize.Bicubic(format=vs.RGBH, range_s="full"), model=model, input_size=input_size, device_index=device_index, num_streams=num_streams)
    else:        
        clipb = vsddcolor.ddcolor(clipb.resize.Bicubic(format=vs.RGBS, range_s="full"), model=model, input_size=input_size, device_index=device_index, num_streams=num_streams) 
    
    # adjusting color space to RGB24 for deoldify
    clipb_rgb = clipb.resize.Bicubic(format=vs.RGB24, range_s="full")
    
    if tweaks_enabled and hue_adjust != 'none':
        clipb_rgb = vs_adjust_clip_hue(clipb_rgb, hue_adjust.lower())
    
    if tweaks_enabled:
        return vs_recover_clip_luma(clip, clipb_rgb)
    else:
        return clipb_rgb
