"""
------------------------------------------------------------------------------- 
Author: Dan64
Date: 2024-02-29
version: 
LastEditors: Dan64
LastEditTime: 2024-04-05
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

from .deoldify.visualize import *
from .deoldify.adjust import *

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*?Your .*? set is empty.*?")
warnings.filterwarnings("ignore", category=UserWarning, message="The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.")
warnings.filterwarnings("ignore", category=FutureWarning, message="Arguments other than a weight enum or `None`.*?")
warnings.filterwarnings("ignore", category=UserWarning, message="Arguments other than a weight enum or `None`.*?")
warnings.filterwarnings("ignore", category=UserWarning, message="torch.nn.utils.weight_norm is deprecated.*?")

__version__ = "3.0.0"

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
wrapper to deoldify() functions with additional filters pre-process and post-process
"""
def ddeoldify(
    clip: vs.VideoNode, method: int = 2, mweight: float = 0.4, deoldify_p: list = [0, 24, 1.0, 0.0], ddcolor_p: list = [1, 24, 1.0, 0.0, True], ddtweak: bool = False, ddtweak_p: list = [0.0, 1.0, 2.5, True, 0.2, 0.5, 1.5, 0.5],  cmc_tresh: float = 0.2, lmm_p: list = [0.2, 0.8, 1.0], alm_p: list = [0.8, 1.0, 0.15], dark: bool = False, dark_p: list = [0.3, 0.8], cmb_sw: bool = False, device_index: int = 0, torch_dir: str = model_dir) -> vs.VideoNode:
    """A Deep Learning based project for colorizing and restoring old images and video using Deoldify and DDColor 

    :param clip:                clip to process, only RGB24 format is supported.
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
    :param cmc_tresh:           chroma_threshold (%), used by: Constrained "Chroma Merge range" [0-1] (0.01=1%)
    :param lmm_p:               parameters for method: "Luma Masked Merge" (see method=4 for a full explanation) 
                                   [0] : luma_mask_limit: luma limit for build the mask used in Luma Masked Merge, range [0-1] (0.01=1%) 
                                   [1] : luma_white_limit: the mask will appliey a gradient till luma_white_limit, range [0-1] (0.01=1%)
                                   [2] : luma_mask_sat: if < 1 the ddcolor dark pixels will substitute with the desaturated deoldify pixels, range [0-1] (0.01=1%)    
    :param alm_p:               parameters for method: "Adaptive Luma Merge" (see method=5 for a full explanation) 
                                   [0] : luma_threshold: threshold for the gradient merge, range [0-1] (0.01=1%) 
                                   [1] : alpha: exponent parameter used for the weight calculation, range [>0] 
                                   [2] : min_weight: min merge weight, range [0-1] (0.01=1%)
    :param dark:                enable/disable darkness filter, range [True,False]                                        
    :param dark_p:              parameters for darken the clip's dark portions, which sometimes are wrongly colored by the color models
                                   [0] : dark_threshold, luma threshold to select the dark area, range [0.1-0.5] (0.01=1%)  
                                   [1] : dark_amount: amount of desaturation to apply to the dark area, range [0-1] 
    :param cmb_sw:              if true switch the clip order in all the combining methods, range [True,False]                                               
    :param device_index:        device ordinal of the GPU, choices: GPU0...GPU7, CPU=99 (default = 0)
    :param torch_dir:           torch hub dir location, default is model directory, if set to None will switch to torch cache dir.
    """
    
    if (not torch.cuda.is_available() and device_index != 99):
        raise vs.Error("ddeoldify: CUDA is not available")

    if not isinstance(clip, vs.VideoNode):
        raise vs.Error("ddeoldify: this is not a clip")

    chroma_resize = True
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
    
    # unpack dark
    darkness_enabled = dark
    dark_threshold = 0.1
    white_threshold = min(max(dark_p[0], dark_threshold), 0.50) 
    dark_sat = min(max(1.1 - dark_p[1], 0.10), 0.80)  
    dark_bright = -min(max(dark_p[1], 0.20), 0.90) #change the sign to reduce the bright        
    
    if os.path.getsize(os.path.join(model_dir, "ColorizeVideo_gen.pth")) == 0:
        raise vs.Error("ddeoldify: model files have not been downloaded.")

    if device_index > 7 and device_index != 99:
        raise vs.Error("ddeoldify: wrong device_index, choices are: GPU0...GPU7, CPU=99")
        
    if ddcolor_rf != 0 and ddcolor_rf not in range(10, 65):
        raise vs.Error("ddeoldify: ddcolor render_factor must be between: 10-64")
            
    if clip.format.id != vs.RGB24:
        # clip not in RGB24 format, it will be converted
        if (clip.format.color_family == "YUV"):
            clip = clip.resize.Bicubic(format=vs.RGB24, matrix_in_s="709", range_s="limited", dither_type="error_diffusion") 
        else:
            clip = clip.resize.Bicubic(format=vs.RGB24, range_s="limited") 
           
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
                    
    clipa = _deoldify(clip, method=method, model=deoldify_model, render_factor=deoldify_rf, package_dir=package_dir) 
    clipb = _ddcolor(clip, method=method, model=ddcolor_model, render_factor=ddcolor_rf, tweaks_enabled=ddtweak, tweaks=ddtweak_p, enable_fp16=ddcolor_enable_fp16, device_index=device_index)             
           
    clip_colored = combine_models(clip_a=clipa, clip_b=clipb, method=method, sat=[deoldify_sat, ddcolor_sat], hue=[deoldify_hue, ddcolor_hue], clipb_weight=merge_weight, CMC_p=cmc_tresh, LMM_p=lmm_p, ALM_p = alm_p, invert_clips=cmb_sw)
    
    if darkness_enabled:
        clip_colored = vs_chroma_bright_tweak(clip_colored, dark_threshold=dark_threshold, white_threshold=white_threshold, dark_sat=dark_sat, dark_bright=dark_bright)  
                
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
def ddeoldify_stabilizer(clip: vs.VideoNode, render_factor: int = 24, smooth: bool = False, smooth_p: list = [0.3, 0.7, 0.9, 0.05], stab: bool = False, stab_p: list = [5, 'A', 1, 15, 0.2, 0.15]) -> vs.VideoNode:
    """Video color stabilization filter, which can be applied to stabilize the chroma components in ddeoldify colored clips. 
        :param clip:                clip to process, only RGB24 format is supported.
        :param render_factor:       render_factor to apply to the filters, the frame size will be reduced to speed-up the filters, 
                                    but the final resolution will be the one of the original clip. If = 0 will be auto selected. 
                                    This approach takes advantage of the fact that human eyes are much less sensitive to
                                    imperfections in chrominance compared to luminance. This means that it is possible to speed-up
                                    the chroma filters and get a great high-resolution result in the end, range: [0, 10-64]                                       
        :param smooth:              enable/disable chroma smoothing, range [True, False]          
        :param smooth_p:            parameters to adjust the saturation and "vibrancy" of the clip.
                                      [0] : dark_threshold, luma threshold to select the dark area, range [0-1] (0.01=1%)  
                                      [1] : white_threshold, if > dark_threshold will be applied a gradient till white_threshold, range [0-1] (0.01=1%)  
                                      [2] : dark_sat, amount of de-saturation to apply to the dark area, range [0-1] 
                                      [3] : dark_bright, darkness parameter it used to reduce the "V" component in "HSV" colorspace, range [0, 1] 
        :param stab:                enable/disable chroma stabilizer, range [True, False]                                        
        :param stab_p:              parameters for the temporal color stabilizer
                                      [0] : nframes, number of frames to be used in the stabilizer, range[3-15]
                                      [1] : mode, type of average used by the stabilizer: range['A'='arithmetic', 'W'='weighted']
                                      [2] : sat: saturation applied to the restored gray prixels [0,1]
                                      [3] : tht, threshold to detect gray pixels, range [0,235], if=0 is not applied the restore,
                                            its value depends on merge method used, suggested values are:
                                                method 0: tht = 5
                                                method 1: tht = 60 (ddcolor provides very saturared frames)
                                                method 2: tht = 15
                                                method 3: tht = 20
                                                method 4: tht = 5
                                                method 5: tht = 10
                                      [4] : weight, weight to blend the restored imaage (default=0.2), range [0-1], if=0 is not applied the blending 
                                      [5] : tht_scen, threshold for scene change detection (default = 0.15), if=0 is not activated, range [0.01-0.50]
    """

    if not isinstance(clip, vs.VideoNode):
        raise vs.Error("ddeoldify_video_stabilizer: this is not a clip")

    if clip.format.id != vs.RGB24:
        # clip not in RGB24 format, it will be converted
        if (clip.format.color_family == "YUV"):
            clip = clip.resize.Bicubic(format=vs.RGB24, matrix_in_s="709", range_s="limited", dither_type="error_diffusion") 
        else:
            clip = clip.resize.Bicubic(format=vs.RGB24, range_s="limited") 
        
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
            
    # unpack chroma_smoothing
    chroma_smoothing_enabled = smooth
    dark_threshold = smooth_p[0]
    white_threshold = smooth_p[1]
    dark_sat = smooth_p[2]
    dark_bright = -smooth_p[3] #change the sign to reduce the bright
    
    # unpack chroma_stabilizer
    colstab_enabled = stab
    colstab_nframes = stab_p[0]
    colstab_mode = stab_p[1]
    colstab_sat = stab_p[2]
    colstab_tht = stab_p[3]
    colstab_weight = stab_p[4]
    colstab_tht_scen = stab_p[5]
    colstab_algo = 0
    
    clip_colored = clip
    
    if chroma_smoothing_enabled:
        clip_colored = vs_chroma_bright_tweak(clip_colored, dark_threshold=dark_threshold, white_threshold=white_threshold, dark_sat=dark_sat, dark_bright=dark_bright) 
        
    if colstab_enabled:
        clip_colored = vs_chroma_stabilizer_ex(clip_colored, nframes=colstab_nframes, mode=colstab_mode, sat=colstab_sat, tht=colstab_tht, weight=colstab_weight, algo=colstab_algo)
        
    if chroma_resize_enabled:
        return _clip_chroma_resize(clip_orig, clip_colored)
    else:
        return clip_colored


"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
wrapper to deoldify. 
"""
def _deoldify(clip: vs.VideoNode, method: int = 2, model: int = 0, render_factor: int = 24, package_dir: str = "") -> vs.VideoNode: 
    
    if method == 1:
        return None
        
    match model:
        case 0:
            colorizer = get_image_colorizer(root_folder=Path(package_dir), artistic=False,isvideo=True) 
        case 1:
            colorizer = get_image_colorizer(root_folder=Path(package_dir), artistic=False,isvideo=False) 
        case 2:
            colorizer = get_image_colorizer(root_folder=Path(package_dir), artistic=True,isvideo=False) 
                    
    def ddeoldify_colorize(n: int, f: vs.VideoFrame, colorizer: ModelImageVisualizer = None, render_factor: int = 24) -> vs.VideoFrame:
        img_orig = frame_to_image(f)
        img_color = colorizer.get_transformed_pil_image(img_orig, render_factor=render_factor, post_process=True)
        return image_to_frame(img_color, f.copy()) 
    
    return clip.std.ModifyFrame(clips=[clip], selector=partial(ddeoldify_colorize, colorizer=colorizer, render_factor=render_factor))         

"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
wrapper to function ddcolor() with tweak pre-process.
"""
def _ddcolor(clip: vs.VideoNode, method: int = 2, model: int = 0, render_factor: int = 24, tweaks_enabled: bool = False, tweaks: list = [0.0, 0.9, 0.7, False, 0.3, 0.3], enable_fp16: bool = True, device_index: int = 0, num_streams: int = 1) -> vs.VideoNode:
    
    if method == 0:
        return None
    else: 
        import vsddcolor
    
    input_size = render_factor * 16
       
    # unpack tweaks
    bright = tweaks[0]
    cont = tweaks[1]
    gamma = tweaks[2]
    luma_constrained_tweak=tweaks[3]
    luma_min = tweaks[4] 
    gamma_luma_min = tweaks[5]
    gamma_alpha = tweaks[6]
    gamma_min = tweaks[7]
    
    if tweaks_enabled:     
        if (luma_constrained_tweak):
            clipb = tweak(clip, bright=bright, cont=cont) # contrast and bright are adjusted before the constrainded luma and gamma
            clipb = constrained_tweak(clipb, luma_min = luma_min, gamma=gamma, gamma_luma_min = gamma_luma_min, gamma_alpha = gamma_alpha, gamma_min=gamma_min)
        else:
            clipb = tweak(clip, bright=bright, cont=cont, gamma=gamma)
    else:
        clipb = clip
    # adjusting clip's color space to RGBH for vsDDColor
    if enable_fp16:
        clipb = vsddcolor.ddcolor(clipb.resize.Bicubic(format=vs.RGBH, range_s="limited"), model=model, input_size=input_size, device_index=device_index, num_streams=num_streams)
    else:        
        clipb = vsddcolor.ddcolor(clipb.resize.Bicubic(format=vs.RGBS, range_s="limited"), model=model, input_size=input_size, device_index=device_index, num_streams=num_streams) 
    
    # adjusting color space to RGB24 for deoldify
    clipb_rgb = clipb.resize.Bicubic(format=vs.RGB24, range_s="limited")
    if tweaks_enabled:
        return vs_recover_clip_luma(clip, clipb_rgb)
    else:
        return clipb_rgb

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
    weight: float = 0.2, tht_scen: float = 0.8, return_mask: bool = False) -> vs.VideoNode:
  
    clip = vs_recover_clip_color(clip=clip, clip_color=clip_color, sat=sat, tht=tht, weight=weight, tht_scen=tht_scen, return_mask=return_mask)
    return clip
