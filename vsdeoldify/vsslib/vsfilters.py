"""
------------------------------------------------------------------------------- 
Author: Dan64
Date: 2024-04-08
version: 
LastEditors: Dan64
LastEditTime: 2024-04-08
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
Library of Vapoursynth filter functions.
"""
import vapoursynth as vs
import math
import numpy as np
import cv2
from PIL import Image
from functools import partial

from .vsutils import *
from .imfilters import *
from .restcolor import *

"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description: 
------------------------------------------------------------------------------- 
Function which try to stabilize the colors of a clip using color temporal stabilizer.
As stabilizer is used the Vapoursynth function "std.AverageFrames()", the mode, can
be: "arithmetic", "weighted"
"""
def vs_clip_color_stabilizer(clip: vs.VideoNode = None, nframes: int = 5, mode: str = "A", scenechange: bool = True) -> vs.VideoNode:

    if nframes%2==0:
        nframes +=1
        
    N = max(3, min(nframes, 15))
    
    match mode:
        case "A" | "arithmetic" | "center":       # for compatibility with version 2.0.0
            weight_list = _build_avg_arithmetic(N) 
        case "W" | "weighted" | "left" | "right": # for compatibility with version 2.0.0
            weight_list = _build_avg_weighted(N)
        case _:
            raise vs.Error("ddeoldify: unknown average method: " + mode)    
    
    #vs.core.log_message(2, "weight_list= " + str(len(weight_list))) 

    # convert the clip format for AverageFrames to YUV    
    clip_yuv = clip.resize.Bicubic(format=vs.YUV444PS, matrix_s="709", range_s="limited")   
    # apply AverageFrames to YUV colorspace      
    clip_yuv = vs.core.std.AverageFrames(clip_yuv, weight_list, scale = 100, scenechange = scenechange, planes=[1,2])  
    # convert the clip format for deoldify to RGB24 
    clip_rgb = clip_yuv.resize.Bicubic(format=vs.RGB24, matrix_in_s="709", range_s="limited", dither_type="error_diffusion") 
        
    return clip_rgb

"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description: 
------------------------------------------------------------------------------- 
Filter which try to stabilize the colors of a clip using color temporal stabilizer.
The stabilization is performed by averaging past/future frames. Since the non matched
areas of past/future frames are gray because is missing in the past/future the color 
information, the filter will fill the gray areas with the pixels of current frames 
(eventually de-saturated with the parameter "sat"). The image restored in this way is
blended with the non restored image using the parameter "weight" (if =0 no blending 
is applied). The gray areas are selected by the threshold parameter "tht". All the pixels
in the HSV color space with "S" < "tht" will be considered gray. If "tht=0" no color
frame restore is applied.
"""
def vs_chroma_stabilizer_ex(clip: vs.VideoNode = None, nframes: int = 5, mode: str = "A", sat: float = 1.0, tht: int = 0, weight: float = 0.5, tht_scen: float = 0.8, hue_adjust: str='none', algo: int = 0) -> vs.VideoNode:

    if tht == 0:
        return vs_clip_color_stabilizer(clip, nframes, mode, scenechange=True)
        
    if nframes%2==0:
        nframes +=1
        
    N = max(3, min(nframes, 15))
    
    match mode:
        case "A" | "arithmetic" | "center":       # for compatibility with version 2.0.0
            weight_list = _build_avg_arithmetic(N) 
        case "W" | "weighted" | "left" | "right": # for compatibility with version 2.0.0
            weight_list = _build_avg_weighted(N)
        case _:
            raise vs.Error("ddeoldify: unknown average method: " + mode)    
    
    #vs.core.log_message(2, "algo= " + str(algo)) 
    
    if algo == 0:
        clip_rgb = _average_clips_ex(clip=clip, weight_list=weight_list, sat=sat, tht=tht, weight=weight, tht_scen=tht_scen, hue_adjust=hue_adjust)
    else:
        clip_rgb = _average_frames_ex(clip=clip, weight_list=weight_list, sat=sat, tht=tht, weight=weight, tht_scen=tht_scen, hue_adjust=hue_adjust)
         
    return clip_rgb

def _build_avg_arithmetic(nframes: int = 5) -> list:
        
    N = nframes
    Nh = round((N-1)/2)    
    Wi = math.trunc(100.0/N)
    
    Wc = 100-(N-1)*Wi
    
    weight_list = list()       
    
    for i in range(0, Nh):
        weight_list.append(Wi)
    weight_list.append(Wc)    
    for i in range(0, Nh):
        weight_list.append(Wi)
    
    return weight_list


def _build_avg_weighted(nframes: int = 5) -> list:
    
    N = nframes
    Nh = round((N-1)/2)
    
    WBase = N*(N+1)*0.5
    
    Wi_scale = 1
    Wc_scale = 2
    
    SumWi = 0
    weight_list = list()       
    for i in range(0, Nh):
        Wi=math.trunc(Wi_scale*100*(i+1)/WBase)
        SumWi += Wi    
        weight_list.append(Wi)
    Wc = 100 - Wc_scale * SumWi
    weight_list.append(Wc)    
    for i in range(0, Nh):
        Wi=math.trunc(Wi_scale*100*(i+1)/WBase)
        weight_list.append(Wi)
    
    return weight_list

"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
Function which try to stabilize the colors of a clip using color temporal stabilizer,
the colors of current frame will be averaged with the ones of previous frames.  
(based on ModifyFrame)
"""
def _average_frames_ex(clip: vs.VideoNode = None, weight_list: list = None, sat: float = 1.0, tht: int = 0, weight: float = 0.2, tht_scen: float = 0.8, hue_adjust: str='none') -> vs.VideoNode:
    
    def smooth_frame(n, f, clip_base: vs.VideoNode = None, weight_list: list = None, sat: float = 1.0, tht: int = 0, weight: float = 0.2, tht_scen: float = 0.8, hue_adjust: str='none') -> vs.VideoFrame:
        max_frames = len(weight_list)
        tot_frames = clip_base.num_frames - 10
        f_out = f.copy()
        if n < max_frames or n > tot_frames:
            return f_out
        img_f = list()
        img_base = frame_to_image(f)
        Nh = round((max_frames-1)/2)
        for i in range(0, Nh):
            Ni = n - (Nh - i)
            img_f.append(restore_color(img_base, frame_to_image(clip_base.get_frame(Ni)), sat, tht, weight, tht_scen, hue_adjust, False))
        img_f.append(img_base)
        for i in range(0, Nh):
            Ni = n + (i+1)
            img_f.append(restore_color(img_base, frame_to_image(clip_base.get_frame(Ni)), sat, tht, weight, tht_scen, hue_adjust, False))
        img_m = _color_temporal_stabilizer(img_f, weight_list)
        return image_to_frame(img_m, f_out)    
    clip = clip.std.ModifyFrame(clips=[clip], selector=partial(smooth_frame, clip_base=clip, weight_list=weight_list, sat=sat, tht=tht, weight=weight, tht_scen=tht_scen, hue_adjust=hue_adjust))
    return clip

"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
Function which try to stabilize the colors of a clip using color temporal stabilizer,
the colors of current frame will be averaged with the ones of previous frames.
(based on AverageFrames)
"""
def _average_clips_ex(clip: vs.VideoNode = None, weight_list: list = None, sat: float = 1.0, tht: int = 0, weight: float = 0.2, tht_scen: float = 0.8, hue_adjust: str='none') -> vs.VideoNode:

    max_frames = len(weight_list)
    clips = list()
    clip_yuv = clip.resize.Bicubic(format=vs.YUV420P8, matrix_s="709", range_s="limited", dither_type="error_diffusion")
    Nh = round((max_frames-1)/2)
    for i in range(0, Nh):
        Ni = -(Nh - i)
        clip_i = vs_get_clip_frame(clip=clip, nframe=Ni)
        clip_i = vs_recover_clip_color(clip=clip_i, clip_color=clip, sat=sat, tht=tht, weight=weight, tht_scen=tht_scen, hue_adjust=hue_adjust, return_mask=False)
        clips.append(clip_i.resize.Bicubic(format=vs.YUV420P8, matrix_s="709", range_s="limited", dither_type="error_diffusion"))
    clips.append(clip_yuv)
    for i in range(0, Nh):
        Ni = i + 1
        clip_i = vs_get_clip_frame(clip=clip, nframe=Ni)
        clip_i = vs_recover_clip_color(clip=clip_i, clip_color=clip, sat=sat, tht=tht, weight=weight, tht_scen=tht_scen, hue_adjust=hue_adjust, return_mask=False)
        clips.append(clip_i.resize.Bicubic(format=vs.YUV420P8, matrix_s="709", range_s="limited", dither_type="error_diffusion"))
    clip_avg = vs.core.std.AverageFrames(clips=clips, weights=weight_list, scale=100, planes=[1,2]) 
    # convert the clip format for deoldify to RGB24 
    clip_rgb = clip_avg.resize.Bicubic(format=vs.RGB24, matrix_in_s="709", range_s="limited", dither_type="error_diffusion") 
    
    return clip_rgb
    
"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
wrapper to function AverageFrames() to get frames fast.
"""    
def vs_get_clip_frame(clip: vs.VideoNode, nframe: int = 0) -> vs.VideoNode:
    
    if nframe == 0:
        return clip
    
    n = abs(nframe)
    
    if n > 15:
        raise vs.Error("ddeoldify: nframe must be between: -15, +15")
    
    weights_list = list()
    
    for i in range(-n, n+1):
        if (i == nframe):
            weights_list.append(100)
        else:
            weights_list.append(0)
    
    vs_format = clip.format.id
    
    # clip converted
    clip_yuv = clip.resize.Bicubic(format=vs.YUV444PS, matrix_s="709", range_s="limited")    
    
    # apply AverageFrames to YUV colorspace      
    clip_yuv = vs.core.std.AverageFrames(clip_yuv, weights_list, scale = 100, scenechange = False, planes=[1,2])  
    
    # convert to the original clip format
    if (clip.format.color_family == "YUV"):
        clip = clip_yuv.resize.Bicubic(format=vs_format)     
    else:
        clip = clip_yuv.resize.Bicubic(format=vs_format, matrix_in_s="709", range_s="limited", dither_type="error_diffusion")
    
    return clip

"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description: 
------------------------------------------------------------------------------- 
wrapper to function restore_color() to restore gray frames.
""" 
def vs_recover_clip_color(clip: vs.VideoNode = None, clip_color: vs.VideoNode = None, sat: float = 1.0, tht: int = 0, 
    weight: float = 0.2, tht_scen: float = 0.8, hue_adjust: str='none', return_mask: bool = False) -> vs.VideoNode:
  
    def color_frame(n, f, sat: float = 1.0, tht: int = 0, weight: float = 0.2, tht_scen: float = 0.8, hue_adjust: str='none', return_mask: bool = False):        
        f_out = f[0].copy()        
        if n < 15:
            return f_out        
        img_gray = frame_to_image(f[0])
        img_color = frame_to_image(f[1])
        img_restored = restore_color(img_color, img_gray, sat, tht, weight, tht_scen, hue_adjust, return_mask)
        return image_to_frame(img_restored, f_out)    
    clip = clip.std.ModifyFrame(clips=[clip, clip_color], selector=partial(color_frame, sat=sat, tht=tht, weight=weight, tht_scen=tht_scen, hue_adjust=hue_adjust, return_mask=return_mask))
    return clip

"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description: 
------------------------------------------------------------------------------- 
wrapper to function restore_color() to restore gray frames.
""" 
def vs_adjust_clip_hue(clip: vs.VideoNode = None, hue_adjust: str='none') -> vs.VideoNode:
  
    def color_frame(n, f, hue_adjust: str='none'):        
        f_out = f.copy()        
        if n < 1:
            return f_out        
        img_color = frame_to_image(f)
        img_restored = adjust_hue_range(img_color, hue_adjust=hue_adjust)
        return image_to_frame(img_restored, f_out)    
    clip = clip.std.ModifyFrame(clips=clip, selector=partial(color_frame, hue_adjust=hue_adjust))
    return clip

"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description: 
------------------------------------------------------------------------------- 
Function to which try to stabilize the chroma of a clip using chroma temporal limiter,
the chroma of current frame will be forced to be inside the range defined by max_deviation  
"""
def vs_chroma_limiter(clip: vs.VideoNode = None, deviation: float = 0.05) -> vs.VideoNode:
    max_deviation = max(min(deviation, 0.5), 0.01)
    def limit_chroma_frame(n, f, clip_base: vs.VideoNode = None, max_deviation: float = 0.05):
        f_out = f.copy()
        if n == 0:
            return f_out
        cur_img = frame_to_image(f)
        prv_img = frame_to_image(clip_base.get_frame(n-1))        
        img_m = _chroma_temporal_limiter(cur_img, prv_img, max_deviation)
        return image_to_frame(img_m, f_out)                
    clip = clip.std.ModifyFrame(clips=[clip], selector=partial(limit_chroma_frame, clip_base=clip, max_deviation=max_deviation))
    return clip
    
def _frame_chroma_stabilizer(clip: vs.VideoNode = None, max_deviation: float = 0.05) -> vs.VideoNode:

    def limit_chroma_frame(n, f, clip_base: vs.VideoNode = None, max_deviation: float = 0.05):
        f_out = f.copy()
        if n == 0:
            return f_out
        cur_img = frame_to_image(f)
        prv_img = frame_to_image(clip_base.get_frame(n-1))        
        img_m = _chroma_temporal_limiter(cur_img, prv_img, max_deviation)
        return image_to_frame(img_m, f_out)                
    clip = clip.std.ModifyFrame(clips=[clip], selector=partial(limit_chroma_frame, clip_base=clip, max_deviation=max_deviation))
    return clip

def _clip_chroma_stabilizer(clip: vs.VideoNode = None, max_deviation: float = 0.05) -> vs.VideoNode:

    def limit_chroma_frame(n, f, clip_base: vs.VideoNode = None, max_deviation: float = 0.05):
        return _frame_chroma_stabilizer(clip_base, max_deviation)                
    clip = clip.std.FrameEval(clip, eval=partial(limit_chroma_frame, clip_base=clip, max_deviation=max_deviation),prop_src=[clip])
    return clip

"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
The the pixels with luma below dark_threshold will be desaturared to level defined
by the dark_sat parameter.
"""
def vs_chroma_bright_tweak(clip: vs.VideoNode = None, black_threshold: float = 0.3, white_threshold: float = 0.6, dark_sat: float = 0.8, dark_bright: float = -0.10, chroma_adjust: str='none') -> vs.VideoNode:      
    
    def merge_frame(n, f, black_limit: float = 0.3, white_limit: float = 0.6, dark_bright: float = -0.10, dark_sat: float = 0.8, chroma_adjust: str='none'):                
        img1 = frame_to_image(f)
        img2 = image_tweak(img1, bright=dark_bright, sat=dark_sat, hue_range=chroma_adjust) 
        if black_limit == white_limit:
            img_m = image_luma_merge(img2, img1, black_limit)
        else:
            img_m = w_image_luma_merge(img2, img1, black_limit, white_limit)
        return image_to_frame(img_m, f.copy())                
    return clip.std.ModifyFrame(clips=clip, selector=partial(merge_frame, black_limit=black_threshold, white_limit=white_threshold, dark_bright=dark_bright, dark_sat=dark_sat, chroma_adjust=chroma_adjust))

"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
Filter used to dark more the dark scenes. The amount of darkness is controlled 
by the parameter dark_amount, while the selected are is controlled by the parameter
dark_threshold.
"""
def vs_dark_tweak(clip: vs.VideoNode = None, dark_threshold: float = 0.3, dark_amount: float = 0.8, dark_hue_adjust: str='none') -> vs.VideoNode:

    d_threshold = 0.1
    d_white_thresh = min(max(dark_threshold, d_threshold), 0.50) 
    d_sat = min(max(1.1 - dark_amount, 0.10), 0.80)  
    d_bright = -min(max(dark_amount, 0.20), 0.90)
    
    def merge_frame(n, f, dark_limit: float = 0.3, white_limit: float = 0.6, dark_bright: float = -0.10, dark_sat: float = 0.8, dark_hue_adjust: str='none'):                
        img1 = frame_to_image(f)
        img2 = image_tweak(img1, bright=dark_bright, sat=dark_sat, hue_range=dark_hue_adjust) 
        if dark_limit == white_limit:
            img_m = image_luma_merge(img2, img1, dark_limit)
        else:
            img_m = w_image_luma_merge(img2, img1, dark_limit, white_limit)
        return image_to_frame(img_m, f.copy())                
    return clip.std.ModifyFrame(clips=clip, selector=partial(merge_frame, dark_limit=d_threshold, white_limit=d_white_thresh, dark_bright=d_bright, dark_sat=d_sat, dark_hue_adjust=dark_hue_adjust))

"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
This function force the average luma of a video clip to don't be below the value
defined by the parameter "luma_min". The function allow to modify the gamma
of the clip if the average luma is below the parameter "gamma_luma_min"  
"""
def constrained_tweak(clip: vs.VideoNode = None, luma_min: float = 0.1, gamma: float = 1, gamma_luma_min: float = 0, gamma_alpha: float = 0, gamma_min: float = 0.5) -> vs.VideoNode:

    def change_frame(n, f, luma_min: float = 0.1, gamma: float = 1, gamma_luma_min: float = 0, gamma_alpha: float = 0, gamma_min: float = 0.5):    
        img = frame_to_image(f)       
        img_m = luma_adjusted_levels(img, luma_min, gamma, gamma_luma_min, gamma_alpha, gamma_min)
        return image_to_frame(img_m, f.copy())                

    clipm = clip.std.ModifyFrame(clips=clip, selector=partial(change_frame, luma_min=luma_min, gamma=gamma, gamma_luma_min=gamma_luma_min, gamma_alpha=gamma_alpha, gamma_min=gamma_min))

    return clipm

"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
This function is an extension of the Tweak() function available in Hybrid with
the possibility to change also the gamma of a video clip.    
"""
def vs_tweak(clip: vs.VideoNode, hue: float = 0, sat: float = 1, bright: float = 0, cont: float = 1, gamma: float = 1, coring: bool = True) -> vs.VideoNode:

    if (hue == 0 and sat == 1 and bright == 0 and cont == 1 and gamma == 1):
        return clip  # non changes
             
    c = vs.core
    
    # convert the format for tewak
    clip = clip.resize.Bicubic(format=vs.YUV444PS, matrix_s="709", range_s="limited")
    
    if (hue != 0 or sat != 1) and clip.format.color_family != vs.GRAY:

        hue = hue * math.pi / 180.0
        hue_sin = math.sin(hue)
        hue_cos = math.cos(hue)

        gray = 128 << (clip.format.bits_per_sample - 8)

        chroma_min = 0
        chroma_max = (2 ** clip.format.bits_per_sample) - 1
        if coring:
            chroma_min = 16 << (clip.format.bits_per_sample - 8)
            chroma_max = 240 << (clip.format.bits_per_sample - 8)

        expr_u = "x {} - {} * y {} - {} * + {} + {} max {} min".format(gray, hue_cos * sat, gray, hue_sin * sat, gray, chroma_min, chroma_max)
        expr_v = "y {} - {} * x {} - {} * - {} + {} max {} min".format(gray, hue_cos * sat, gray, hue_sin * sat, gray, chroma_min, chroma_max)

        if clip.format.sample_type == vs.FLOAT:
            expr_u = "x {} * y {} * + -0.5 max 0.5 min".format(hue_cos * sat, hue_sin * sat)
            expr_v = "y {} * x {} * - -0.5 max 0.5 min".format(hue_cos * sat, hue_sin * sat)

        src_u = clip.std.ShufflePlanes(planes=1, colorfamily=vs.GRAY)
        src_v = clip.std.ShufflePlanes(planes=2, colorfamily=vs.GRAY)

        dst_u = c.std.Expr(clips=[src_u, src_v], expr=expr_u)
        dst_v = c.std.Expr(clips=[src_u, src_v], expr=expr_v)

        clip = c.std.ShufflePlanes(clips=[clip, dst_u, dst_v], planes=[0, 0, 0], colorfamily=clip.format.color_family)

    if bright != 0 or cont != 1:

        if clip.format.sample_type == vs.INTEGER:
            luma_lut = []

            luma_min = 0
            luma_max = (2 ** clip.format.bits_per_sample) - 1
            if coring:
                luma_min = 16 << (clip.format.bits_per_sample - 8)
                luma_max = 235 << (clip.format.bits_per_sample - 8)

            for i in range(2 ** clip.format.bits_per_sample):
                val = int((i - luma_min) * cont + bright + luma_min + 0.5)
                luma_lut.append(min(max(val, luma_min), luma_max))

            clip = clip.std.Lut(planes=0, lut=luma_lut)
        else:
            expression = "x {} * {} + 0.0 max 1.0 min".format(cont, bright)

            clip = clip.std.Expr(expr=[expression, "", ""])
            
    # convert the clip format for deoldify and std.Levels() to RGB24 
    clip_rgb = clip.resize.Bicubic(format=vs.RGB24, matrix_in_s="709", range_s="limited", dither_type="error_diffusion") 
    
    if gamma != 1:
        clip_rgb = clip_rgb.std.Levels(gamma=gamma) 
    
    return clip_rgb
 

"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
Function to copy the luma of video Clip "orig" in the video "clip" 
"""
def vs_recover_clip_luma(orig: vs.VideoNode = None, clip: vs.VideoNode = None) -> vs.VideoNode:
    def copy_luma_frame(n, f):    
        img_orig = frame_to_image(f[0])
        img_clip = frame_to_image(f[1])        
        img_m = chroma_post_process(img_clip, img_orig)
        return image_to_frame(img_m, f[0].copy())                
    clip = clip.std.ModifyFrame(clips=[orig, clip], selector=copy_luma_frame)
    return clip


"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description: ONLY FOR TESTING
------------------------------------------------------------------------------- 
This filter return a mask calculated on luma range..
"""
def vs_luma_mask(clip: vs.VideoNode = None, luma_mask_limit: float = 0.4, luma_white_limit: float = 0.7) -> vs.VideoNode:
   
    def mask_frame(n, f, luma_limit: float = 0.4, white_limit: float = 0.7):                
        img1 = frame_to_image(f)
        if luma_limit == white_limit:
            #vs.core.log_message(2, "frame[" + str(n) + "]: luma_limit = " + str(luma_limit)) 
            img_masked = image_luma_merge(img1, img1, luma_limit, True)
        else:
            img_masked = w_image_luma_merge(img1, img1, luma_limit, white_limit, True)
        return image_to_frame(img_masked, f.copy())                
    clipm = clip.std.ModifyFrame(clips=clip, selector=partial(mask_frame, luma_limit = luma_mask_limit, white_limit = luma_white_limit))
    return clipm    

"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description: ONLY FOR TESTING
------------------------------------------------------------------------------- 
Vapoursynth version of AdaptiveLumaMerge (very slow).
"""
def vs_adaptive_Merge(clipa: vs.VideoNode = None, clipb: vs.VideoNode = None, clipb_weight: float = 0.0) -> vs.VideoNode:
    #Vapoursynth version
    def merge_frame(n, f, core, clipa: vs.VideoNode = None, clipb: vs.VideoNode = None, clipb_weight: float = 0.0):               
        clip1 = clipa[n]
        clip2 = clipb[n] 
        clip2_yuv = clip2.resize.Bicubic(format=vs.YUV444PS, matrix_s="709", range_s="limited") 
        clip2_avg_y = vs.core.std.PlaneStats(clip2_yuv, plane=0)
        luma = clip2_avg_y.get_frame(0).props['PlaneStatsAverage']
        #vs.core.log_message(2, "Luma(" + str(n) + ") = " + str(luma))
        brightness = min(1.5 * luma, 1)
        w = max(clipb_weight * brightness, 0.15)
        clip3 = core.std.Merge(clip1, clip2, weight=w)  
        f_out = f.copy()
        f_out = clip3.get_frame(0)
        return f_out
    clipm = clipa.std.ModifyFrame(clips=clipa, selector=partial(merge_frame, core=vs.core, clipa=clipa, clipb=clipb, clipb_weight=clipb_weight))
    return clipm
