"""
------------------------------------------------------------------------------- 
Author: Dan64
Date: 2024-02-29
version: 
LastEditors: Dan64
LastEditTime: 2024-05-11
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
main wrapper for Vapoursynth ddcolor() filer
"""
from __future__ import annotations
from functools import partial

import os

os.environ["CUDA_MODULE_LOADING"] = "LAZY"
os.environ["NUMEXPR_MAX_THREADS"] = "8"

import math 
from .deoldify import device
from .deoldify.device_id import DeviceId

from .vsslib.vsfilters import *
from .vsslib.mcomb import *
from .vsslib.vsmodels import *

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*?Your .*? set is empty.*?")
warnings.filterwarnings("ignore", category=UserWarning, message="The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.")
warnings.filterwarnings("ignore", category=FutureWarning, message="Arguments other than a weight enum or `None`.*?")
warnings.filterwarnings("ignore", category=UserWarning, message="Arguments other than a weight enum or `None`.*?")
warnings.filterwarnings("ignore", category=UserWarning, message="torch.nn.utils.weight_norm is deprecated.*?")

__version__ = "3.5.4"

package_dir = os.path.dirname(os.path.realpath(__file__))
model_dir = os.path.join(package_dir, "models")

#configuring torch
torch.backends.cudnn.benchmark=True

import vapoursynth as vs

"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
wrapper to deoldify() function with "presets" management
"""
def ddeoldify_main(clip: vs.VideoNode, Preset: str = 'Fast', VideoTune: str = 'Stable', ColorFix: str = 'Violet/Red', ColorTune: str = 'Light', ColorMap: str = 'None', degrain_strength: int = 0, enable_fp16: bool = True) -> vs.VideoNode:
    """Main vsdeoldify function supporting the Presets
    
    :param clip:                clip to process, only RGB24 format is supported.                                
    :param Preset:              Preset to control the encoding speed/quality.
                                Allowed values are:
                                    'Placebo', 
                                    'VerySlow', 
                                    'Slower', 
                                    'Slow', 
                                    'Medium', 
                                    'Fast',  (default)
                                    'Faster', 
                                    'VeryFast'                                    
    :param VideoTune:           Preset to control the output video color stability
                                Allowed values are:
                                    'VeryStable', 
                                    'MoreStable'
                                    'Stable', 
                                    'Balanced', 
                                    'Vivid', 
                                    ,MoreVivid'
                                    'VeryVivid',                                
    :param ColorFix:            This parameter allows to reduce color noise on determinated chroma ranges.
                                Allowed values are:
                                    'None',  
                                    'Magenta', 
                                    'Magenta/Violet', 
                                    'Violet', 
                                    'Violet/Red', (default) 
                                    'Blue/Magenta', 
                                    'Yellow', 
                                    'Yellow/Orange', 
                                    'Yellow/Green'
    :param ColorTune:           This parameter allows to define the intensity of noise reduction applied by ColorFix.
                                Allowed values are:
                                    'Light',  (default)
                                    'Medium', 
                                    'Strong', 
    :param ColorMap:            This parameter allows to change a given color range to another color. 
                                Allowed values are:
                                    'None', (default)
                                    'Blue->Brown', 
                                    'Blue->Red', 
                                    'Blue->Green', 
                                    'Green->Brown', 
                                    'Green->Red', 
                                    'Green->Blue', 
                                    'Red->Brown', 
                                    'Red->Blue'
                                    'Yellow->Rose'                                     
    :param degrain_strength:    strenght of denoise/degrain pre-filter applied on BW clip, if = 0 the pre-filter is disabled, range [0-5], default = 0                                
    :param enable_fp16:         Enable/disable FP16 in ddcolor inference, range [True, False]                                    
    """
    # Select presets / tuning
    Preset = Preset.lower()
    presets = ['placebo', 'veryslow', 'slower', 'slow', 'medium', 'fast', 'faster', 'veryfast']
    preset0_rf = [34, 32, 30, 28, 26, 24, 20, 16]
    preset1_rf = [48, 44, 36, 32, 28, 24, 20, 16]
    
    try:
        pr_id = presets.index(Preset)
    except ValueError:
        raise vs.Error("ddeoldify: Preset choice is invalid for '" + pr_id + "'")
    
    deoldify_rf = preset0_rf[pr_id]
    ddcolor_rf = preset1_rf[pr_id]
    
    #vs.core.log_message(2, "Preset index: " + str(pr_id) )

    # Select VideoTune
    VideoTune = VideoTune.lower()
    video_tune = ['verystable', 'morestable', 'stable', 'balanced', 'vivid', 'morevivid', 'veryvivid' ]  
    ddcolor_weight = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]   

    try:
        w_id = video_tune.index(VideoTune)
    except ValueError:
        raise vs.Error("ddeoldify: VideoTune choice is invalid for '" + VideoTune + "'")    
    
    # Select ColorTune for ColorFix
    ColorTune = ColorTune.lower()
    color_tune = ['light', 'medium', 'strong']  
    hue_tune = ["0.8,0.1", "0.5,0.1", "0.2,0.1"]       
    hue_tune2 = ["0.9,0", "0.7,0", "0.5,0"]       
    
    try:
        tn_id = color_tune.index(ColorTune)
    except ValueError:
        raise vs.Error("ddeoldify: ColorTune choice is invalid for '" + ColorTune + "'")
            
    # Select ColorFix for ddcolor/stabilizer
    ColorFix = ColorFix.lower()
    color_fix = ['none', 'magenta', 'magenta/violet', 'violet', 'violet/red', 'blue/magenta', 'yellow', 'yellow/orange', 'yellow/green']  
    hue_fix = ["none", "270:300", "270:330", "300:330", "300:360", "220:280" , "60:90", "30:90", "60:120"]       

    try:
        co_id = color_fix.index(ColorFix)
    except ValueError:
        raise vs.Error("ddeoldify: ColorFix choice is invalid for '" + ColorFix + "'")
    
    if co_id == 0:
        hue_range = "none"
        hue_range2 = "none"
    else:    
        hue_range = hue_fix[co_id] + "|" + hue_tune[tn_id]
        hue_range2 = hue_fix[co_id] + "|" + hue_tune2[tn_id]
        
    # Select Color Mapping
    ColorMap = ColorMap.lower()    
    colormap = ['none', 'blue->brown', 'blue->red', 'blue->green', 'green->brown', 'green->red', 'green->blue', 'red->brown', 'red->blue', 'yellow->rose']
    hue_map = ["none", "180:280|+140,0.4", "180:280|+100,0.4", "180:280|+220,0.4", "80:180|+260,0.4", "80:180|+220,0.4", "80:180|+140,0.4", "300:360,0:20|+40,0.6", "300:360,0:20|+260,0.6", "30:90|+300,0.8"]
        
    try:
        cl_id = colormap.index(ColorMap)
    except ValueError:
        raise vs.Error("ddeoldify: ColorMap choice is invalid for '" + ColorMap + "'")
    
    chroma_adjust = hue_map[cl_id] 
    
    clip_colored = ddeoldify(clip, method=2, mweight=ddcolor_weight[w_id], deoldify_p=[0, deoldify_rf, 1.0, 0.0], ddcolor_p=[1, ddcolor_rf, 1.0, 0.0, enable_fp16], ddtweak=True, ddtweak_p=[0.0, 1.0, 2.5, True, 0.3, 0.6, 1.5, 0.5, hue_range], degrain_strength=degrain_strength)              
    
    if pr_id > 5 and cl_id > 0:    
        clip_colored = ddeoldify_stabilizer(clip_colored, colormap=chroma_adjust)
    elif pr_id > 3:       
        clip_colored = ddeoldify_stabilizer(clip_colored, dark=True, dark_p= [0.2, 0.8], smooth=True, smooth_p=[0.3, 0.7, 0.9, 0.0, chroma_adjust], stab=True, stab_p=[5, 'A', 1, 15, 0.2, 0.15])
    else:
        clip_colored = ddeoldify_stabilizer(clip_colored, dark=True, dark_p= [0.2, 0.8], smooth=True, smooth_p=[0.3, 0.7, 0.9, 0.0, chroma_adjust], stab=True, stab_p=[5, 'A', 1, 15, 0.2, 0.15, hue_range2])
        
    return clip_colored

"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
wrapper to deoldify() functions with additional filters pre-process and post-process
"""
def ddeoldify(
    clip: vs.VideoNode, method: int = 2, mweight: float = 0.4, deoldify_p: list = [0, 24, 1.0, 0.0], ddcolor_p: list = [1, 24, 1.0, 0.0, True], dotweak: bool = False, dotweak_p: list = [0.0, 1.0, 1.0, False, 0.2, 0.5, 1.5, 0.5], ddtweak: bool = False, ddtweak_p: list = [0.0, 1.0, 2.5, True, 0.3, 0.6, 1.5, 0.5, "300:360|0.8,0.1"],  degrain_strength: int = 0, cmc_tresh: float = 0.2, lmm_p: list = [0.2, 0.8, 1.0], alm_p: list = [0.8, 1.0, 0.15], cmb_sw: bool = False, device_index: int = 0, torch_dir: str = model_dir) -> vs.VideoNode:
    """A Deep Learning based project for colorizing and restoring old images and video using Deoldify and DDColor 

    :param clip:                clip to process, only RGB24 format is supported                                 
    :param method:              method used to combine deoldify() with ddcolor() (default = 2): 
                                    0 : deoldify only (no merge)
                                    1 : ddcolor only (no merge)                                 
                                    2 : Simple Merge (default): 
                                        the frames are combined using a weighted merge, where the parameter "mweight"
                                        represent the weight assigned to the colors provided by the ddcolor() frames.
                                        With this method is suggested a starting weight < 50% (ex. = 40%).                                         
                                    3 : Constrained Chroma Merge:
                                        given that the colors provided by deoldify() are more conservative and stable 
                                        than the colors obtained with ddcolor(). The frames are combined by assigning
                                        a limit to the amount of difference in chroma values between deoldify() and
                                        ddcolor() this limit is defined by the threshold parameter "cmc_tresh". 
                                        The limit is applied to the image converted to "YUV". For example when 
                                        cmc_tresh=0.2, the chroma values "U","V" of ddcolor() frame will be constrained 
                                        to have an absolute percentage difference respect to "U","V" provided by deoldify()
                                        not higher than 20%. The final limited frame will be merged again with the deoldify()
                                        frame. With this method is suggested a starting weight > 50% (ex. = 60%).                                         
                                    4 : Luma Masked Merge:   
                                        the frames are combined using a masked merge, the pixels of ddcolor() with luma < "luma_mask_limit"
                                        will be filled with the pixels of deoldify(). If "luma_white_limit" > "luma_mask_limit" the mask will 
                                        apply a gradient till "luma_white_limit". If the parameter "mweight" > 0 the final masked frame will 
                                        be merged again with the deoldify() frame. With this method is suggested a starting weight > 60%
                                        (ex. = 70%).
                                    5 : Adaptive Luma Merge:
                                        given that the ddcolor() perfomance is quite bad on dark scenes, the images are 
                                        combined by decreasing the weight assigned to ddcolor() when the luma is 
                                        below a given threshold given by: luma_threshold. The weight is calculated using
                                        the formula: merge_weight = max(mweight * (luma/luma_threshold)^alpha, min_weight).
                                        For example with: luma_threshold = 0.6 and alpha = 1, the weight assigned to 
                                        ddcolor() will start to decrease linearly when the luma < 60% till "min_weight".
                                        For alpha=2, begins to decrease quadratically (because luma/luma_threshold < 1).
                                        With this method is suggested a starting weight > 70% (ex. = 80%).                                         
                                    The methods 3 and 4 are similar to Simple Merge, but before the merge with deoldify()
                                    the ddcolor() frame is limited in the chroma changes (method 3) or limited based on the luma
                                    (method 4). The method 5 is a Simple Merge where the weight decrease with luma.         
    :param mweight:             weight given to ddcolor's clip in all merge methods, range [0-1] (0.01=1%) 
    :param deoldify_p:          parameters for deoldify():            
                                   [0] deoldify model to use (default = 0):
                                       0 = ColorizeVideo_gen
                                       1 = ColorizeStable_gen
                                       2 = ColorizeArtistic_gen
                                   [1] render factor for the model, range: 10-44 (default = 24).
                                   [2] saturation parameter to apply to deoldify color model (default = 1)
                                   [3] hue parameter to apply to deoldify color model (default = 0)
    :param ddcolor_p:           parameters for ddcolor(): 
                                   [0] ddcolor model (default = 1): 
                                       0 = ddcolor_modelscope, 
                                       1 = ddcolor_artistic
                                   [1] render factor for the model, if=0 will be auto selected 
                                       (default = 24), range: [0, 10-64] 
                                   [2] saturation parameter to apply to deoldify color model (default = 1)
                                   [3] hue parameter to apply to deoldify color model (default = 0)
                                   [4] enable/disable FP16 in ddcolor inference
    :param dotweak:             enabled/disable tweak parameters for deoldify(), range [True,False]                                   
    :param dotweak_p:           tweak parameters for ddeoldify():                                   
                                   [0] : ddcolor tweak's bright (default = 0)
                                   [1] : ddcolor tweak's constrast (default = 1), if < 1 ddeoldify provides de-saturated frames
                                   [2] : ddcolor tweak's gamma (default = 1) 
                                   [3] : luma constrained gamma -> luma constrained gamma correction enabled (default = False), range: [True, False]
                                            When enabled the average luma of a video clip will be forced to don't be below the value
                                            defined by the parameter "luma_min". The function allow to modify the gamma
                                            of the clip if the average luma is below the parameter "gamma_luma_min". A gamma value > 2.0 improves 
                                            the ddcolor stability on bright scenes, while a gamma < 1 improves the ddcolor stability on 
                                            dark scenes. The decrease of the gamma with luma is activated using a gamma_alpha != 0.                                                
                                   [4] : luma_min: luma (%) min value for tweak activation (default = 0.2), if=0 is not activated, range [0-1]
                                   [5] : gamma_luma_min: luma (%) min value for gamma tweak activation (default = 0.5), if=0 is not activated, range [0-1]
                                   [6] : gamma_alpha: the gamma will decrease with the luma g = max(gamma * pow(luma/gamma_luma_min, gamma_alpha), gamma_min),
                                         for a movie with a lot of dark scenes is suggested alpha > 1, if=0 is not activated, range [>=0]
                                   [7] : gamma_min: minimum value for gamma, range (default=0.5) [>0.1]
                                   [8] : "chroma adjustment" parameter (optional), if="none" is disabled (see the README) 
    :param ddtweak:             enabled/disable tweak parameters for ddcolor(), range [True,False]                                   
    :param ddtweak_p:           tweak parameters for ddcolor():                                   
                                   [0] : ddcolor tweak's bright (default = 0)
                                   [1] : ddcolor tweak's constrast (default = 1), if < 1 ddcolor provides de-saturated frames
                                   [2] : ddcolor tweak's gamma (default = 1) 
                                   [3] : luma constrained gamma -> luma constrained gamma correction enabled (default = False), range: [True, False]
                                            When enabled the average luma of a video clip will be forced to don't be below the value
                                            defined by the parameter "luma_min". The function allow to modify the gamma
                                            of the clip if the average luma is below the parameter "gamma_luma_min". A gamma value > 2.0 improves 
                                            the ddcolor stability on bright scenes, while a gamma < 1 improves the ddcolor stability on 
                                            dark scenes. The decrease of the gamma with luma is activated using a gamma_alpha != 0.                                                
                                   [4] : luma_min: luma (%) min value for tweak activation (default = 0.2), if=0 is not activated, range [0-1]
                                   [5] : gamma_luma_min: luma (%) min value for gamma tweak activation (default = 0.5), if=0 is not activated, range [0-1]
                                   [6] : gamma_alpha: the gamma will decrease with the luma g = max(gamma * pow(luma/gamma_luma_min, gamma_alpha), gamma_min),
                                         for a movie with a lot of dark scenes is suggested alpha > 1, if=0 is not activated, range [>=0]
                                   [7] : gamma_min: minimum value for gamma, range (default=0.5) [>0.1]
                                   [8] : "chroma adjustment" parameter (optional), if="none" is disabled (see the README) 
    :param degrain_strength:    strenght of denoise/degrain pre-filter applied on BW clip, if = 0 the pre-filter is disabled, range [0-5], default = 0 
    :param cmc_tresh:           chroma_threshold (%), used by: Constrained "Chroma Merge range" [0-1] (0.01=1%)
    :param lmm_p:               parameters for method: "Luma Masked Merge" (see method=4 for a full explanation) 
                                   [0] : luma_mask_limit: luma limit for build the mask used in Luma Masked Merge, range [0-1] (0.01=1%) 
                                   [1] : luma_white_limit: the mask will appliey a gradient till luma_white_limit, range [0-1] (0.01=1%)
                                   [2] : luma_mask_sat: if < 1 the ddcolor dark pixels will substitute with the desaturated deoldify pixels, range [0-1] (0.01=1%)    
    :param alm_p:               parameters for method: "Adaptive Luma Merge" (see method=5 for a full explanation) 
                                   [0] : luma_threshold: threshold for the gradient merge, range [0-1] (0.01=1%) 
                                   [1] : alpha: exponent parameter used for the weight calculation, range [>0] 
                                   [2] : min_weight: min merge weight, range [0-1] (0.01=1%)    
    :param cmb_sw:              if true switch the clip order in all the combining methods, range [True,False]                                               
    :param device_index:        device ordinal of the GPU, choices: GPU0...GPU7, CPU=99 (default = 0)
    :param torch_dir:           torch hub dir location, default is model directory, if set to None will switch to torch cache dir.
    """
    
    if (not torch.cuda.is_available() and device_index != 99):
        raise vs.Error("ddeoldify: CUDA is not available")

    if not isinstance(clip, vs.VideoNode):
        raise vs.Error("ddeoldify: this is not a clip")
    
    chroma_resize = True  # always enabled
    merge_weight = mweight
    
    # unpack deoldify_params
    deoldify_model = deoldify_p[0]
    deoldify_rf = deoldify_p[1] 
    deoldify_sat = deoldify_p[2]
    deoldify_hue = deoldify_p[3]
    
    # unpack deoldify_params
    ddcolor_model = ddcolor_p[0]
    ddcolor_rf = ddcolor_p[1] 
    ddcolor_sat = ddcolor_p[2]
    ddcolor_hue = ddcolor_p[3]
    ddcolor_enable_fp16 = ddcolor_p[4]       
    
    if os.path.getsize(os.path.join(model_dir, "ColorizeVideo_gen.pth")) == 0:
        raise vs.Error("ddeoldify: model files have not been downloaded.")

    if device_index > 7 and device_index != 99:
        raise vs.Error("ddeoldify: wrong device_index, choices are: GPU0...GPU7, CPU=99")
        
    if ddcolor_rf != 0 and ddcolor_rf not in range(10, 65):
        raise vs.Error("ddeoldify: ddcolor render_factor must be between: 10-64")
            
    if clip.format.id != vs.RGB24:
        # clip not in RGB24 format, it will be converted
        if (clip.format.color_family == "YUV"):
            clip = clip.resize.Bicubic(format=vs.RGB24, matrix_in_s="709", range_s="full", dither_type="error_diffusion") 
        else:
            clip = clip.resize.Bicubic(format=vs.RGB24, range_s="full") 
           
    #choices: GPU0...GPU7, CPU=99 
    device.set(device=DeviceId(device_index))
    
    if torch_dir != None:
        torch.hub.set_dir(torch_dir)      
         
    if ddcolor_rf == 0:
        ddcolor_rf = min(max(math.trunc(0.4 * clip.width / 16), 16), 48)    
         
    clipb_weight = merge_weight
    
    if chroma_resize:
        frame_size = min(max(ddcolor_rf, deoldify_rf) * 16, clip.width)     # frame size calculation for inference()  
        clip_orig = clip;
        clip = clip.resize.Spline64(width=frame_size, height=frame_size)        
    
    clipa = vs_deoldify(clip, method=method, model=deoldify_model, render_factor=deoldify_rf, tweaks_enabled=dotweak, tweaks=dotweak_p, package_dir=package_dir) 
    clipb = vs_ddcolor(clip, method=method, model=ddcolor_model, render_factor=ddcolor_rf, tweaks_enabled=ddtweak, tweaks=ddtweak_p, dstrength=degrain_strength, enable_fp16=ddcolor_enable_fp16, device_index=device_index)             
           
    clip_colored = vs_combine_models(clip_a=clipa, clip_b=clipb, method=method, sat=[deoldify_sat, ddcolor_sat], hue=[deoldify_hue, ddcolor_hue], clipb_weight=merge_weight, CMC_p=cmc_tresh, LMM_p=lmm_p, ALM_p = alm_p, invert_clips=cmb_sw)
    
    if chroma_resize:
        return _clip_chroma_resize(clip_orig, clip_colored)
    else:
        return clip_colored

"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
Video color stabilization filter, derived from ddeoldify
"""
def ddeoldify_stabilizer(clip: vs.VideoNode, dark: bool = False, dark_p: list = [0.2, 0.8], smooth: bool = False, smooth_p: list = [0.3, 0.7, 0.9, 0.0, "none"], stab: bool = False, stab_p: list = [5, 'A', 1, 15, 0.2, 0.15], colormap: str = "none", render_factor: int = 24) -> vs.VideoNode:
    """Video color stabilization filter, which can be applied to stabilize the chroma components in ddeoldify colored clips. 
        :param clip:                clip to process, only RGB24 format is supported.
        :param dark:                enable/disable darkness filter, range [True,False]                                        
        :param dark_p:              parameters for darken the clip's dark portions, which sometimes are wrongly colored by the color models
                                      [0] : dark_threshold, luma threshold to select the dark area, range [0.1-0.5] (0.01=1%)  
                                      [1] : dark_amount: amount of desaturation to apply to the dark area, range [0-1]   
                                      [2] : "chroma range" parameter (optional), if="none" is disabled (see the README)                                       
        :param smooth:              enable/disable chroma smoothing, range [True, False]          
        :param smooth_p:            parameters to adjust the saturation and "vibrancy" of the clip.
                                      [0] : dark_threshold, luma threshold to select the dark area, range [0-1] (0.01=1%)  
                                      [1] : white_threshold, if > dark_threshold will be applied a gradient till white_threshold, range [0-1] (0.01=1%)  
                                      [2] : dark_sat, amount of de-saturation to apply to the dark area, range [0-1] 
                                      [3] : dark_bright, darkness parameter it used to reduce the "V" component in "HSV" colorspace, range [0, 1] 
                                      [4] : "chroma range" parameter (optional), if="none" is disabled (see the README)  
        :param stab:                enable/disable chroma stabilizer, range [True, False]                                        
        :param stab_p:              parameters for the temporal color stabilizer
                                      [0] : nframes, number of frames to be used in the stabilizer, range[3-15]
                                      [1] : mode, type of average used by the stabilizer: range['A'='arithmetic', 'W'='weighted']
                                      [2] : sat: saturation applied to the restored gray prixels [0,1]
                                      [3] : tht, threshold to detect gray pixels, range [0,255], if=0 is not applied the restore,
                                            its value depends on merge method used, suggested values are:
                                                method 0: tht = 5
                                                method 1: tht = 60 (ddcolor provides very saturared frames)
                                                method 2: tht = 15
                                                method 3: tht = 20
                                                method 4: tht = 5
                                                method 5: tht = 10
                                      [4] : weight, weight to blend the restored imaage (default=0.2), range [0-1], if=0 is not applied the blending 
                                      [5] : tht_scen, threshold for scene change detection (default = 0.15), if=0 is not activated, range [0.01-0.50]
                                      [6] : "chroma adjustment" parameter (optional), if="none" is disabled (see the README) 
        :param colormap             direct hue/color mapping, without luma filtering, using the "chroma adjustment" parameter, if="none" is disabled 
        :param render_factor:       render_factor to apply to the filters, the frame size will be reduced to speed-up the filters, 
                                    but the final resolution will be the one of the original clip. If = 0 will be auto selected. 
                                    This approach takes advantage of the fact that human eyes are much less sensitive to
                                    imperfections in chrominance compared to luminance. This means that it is possible to speed-up
                                    the chroma filters and get a great high-resolution result in the end, range: [0, 10-64]            
    """

    if not isinstance(clip, vs.VideoNode):
        raise vs.Error("ddeoldify_video_stabilizer: this is not a clip")

    if clip.format.id != vs.RGB24:
        # clip not in RGB24 format, it will be converted
        if (clip.format.color_family == "YUV"):
            clip = clip.resize.Bicubic(format=vs.RGB24, matrix_in_s="709", range_s="full", dither_type="error_diffusion") 
        else:
            clip = clip.resize.Bicubic(format=vs.RGB24, range_s="full") 
        
    # enable chroma_resize
    chroma_resize_enabled = True
    
    if render_factor != 0 and render_factor not in range(16, 65):
        raise vs.Error("ddeoldify_stabilizer: render_factor must be between: 16-64")
        
    if render_factor == 0:
        render_factor = min(max(math.trunc(0.4 * clip.width / 16), 16), 64)
                
    if chroma_resize_enabled:
        frame_size = min(render_factor * 16, clip.width) # frame size calculation for filters
        clip_orig = clip;
        clip = clip.resize.Spline64(width=frame_size, height=frame_size) 
    
    # unpack dark
    dark_enabled = dark
    dark_threshold = dark_p[0]
    dark_amount = dark_p[1] 
    if (len(dark_p) > 2): 
        dark_hue_adjust = dark_p[2]
    else:
        dark_hue_adjust = 'none'
        
    # unpack chroma_smoothing
    chroma_smoothing_enabled = smooth
    black_threshold = smooth_p[0]
    white_threshold = smooth_p[1]
    dark_sat = smooth_p[2]
    dark_bright = -smooth_p[3] #change the sign to reduce the bright
    if (len(smooth_p) > 4): 
        chroma_adjust = smooth_p[4]
    else:
        chroma_adjust = 'none'

    # define colormap
    colormap = colormap.lower()
    colormap_enabled = (colormap != "none" and colormap != "")    
    
    # unpack chroma_stabilizer
    stab_enabled = stab
    stab_nframes = stab_p[0]
    stab_mode = stab_p[1]
    stab_sat = stab_p[2]
    stab_tht = stab_p[3]
    stab_weight = stab_p[4]
    stab_tht_scen = stab_p[5]
    if len(stab_p) > 6:
        stab_hue_adjust = stab_p[6]
    else:
        stab_hue_adjust = 'none'
    stab_algo = 0
    
    clip_colored = clip
    
    if dark_enabled:
        clip_colored = vs_dark_tweak(clip_colored, dark_threshold=dark_threshold, dark_amount=dark_amount, dark_hue_adjust=dark_hue_adjust.lower())  
                    
    if chroma_smoothing_enabled:
        clip_colored = vs_chroma_bright_tweak(clip_colored, black_threshold=black_threshold, white_threshold=white_threshold, dark_sat=dark_sat, dark_bright=dark_bright, chroma_adjust=chroma_adjust.lower()) 
    
    if colormap_enabled:
        clip_colored = vs_colormap(clip_colored, colormap=colormap) 
    
    if stab_enabled:
        clip_colored = vs_chroma_stabilizer_ex(clip_colored, nframes=stab_nframes, mode=stab_mode, sat=stab_sat, tht=stab_tht, weight=stab_weight, hue_adjust=stab_hue_adjust.lower(), algo=stab_algo)
        
    if chroma_resize_enabled:
        return _clip_chroma_resize(clip_orig, clip_colored)
    else:
        return clip_colored

"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description: ONLY FOR TESTING
------------------------------------------------------------------------------- 
wrapper to function vs_recover_clip_luma().
"""
def _clip_chroma_resize(clip_hires: vs.VideoNode, clip_lowres: vs.VideoNode) -> vs.VideoNode:
    clip_resized = clip_lowres.resize.Spline64(width=clip_hires.width, height=clip_hires.height) 
    return vs_recover_clip_luma(clip_hires, clip_resized)

"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description: ONLY FOR TESTING
------------------------------------------------------------------------------- 
wrapper to function vs_get_clip_frame() to get frames fast.
"""
def _get_clip_frame(clip: vs.VideoNode, nframe: int = 0) -> vs.VideoNode:
    
    clip = vs_get_clip_frame(clip=clip, nframe=nframe)
    return clip    

"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description: ONLY FOR TESTING
------------------------------------------------------------------------------- 
wrapper to function vs_recover_clip_color() to restore gray frames.
""" 
def _recover_clip_color(clip: vs.VideoNode = None, clip_color: vs.VideoNode = None, sat: float = 1.0, tht: int = 0, 
    weight: float = 0.2, tht_scen: float = 0.8, hue_adjust: str='none', return_mask: bool = False) -> vs.VideoNode:
  
    clip = vs_recover_clip_color(clip=clip, clip_color=clip_color, sat=sat, tht=tht, weight=weight, tht_scen=tht_scen, hue_adjust=hue_adjust, return_mask=return_mask)
    return clip
