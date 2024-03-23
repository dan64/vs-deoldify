"""
------------------------------------------------------------------------------- 
Author: Dan64
Date: 2024-02-29
version: 
LastEditors: Dan64
LastEditTime: 2024-03-23
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
from vsddcolor import ddcolor

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*?Your .*? set is empty.*?")
warnings.filterwarnings("ignore", category=UserWarning, message="The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.")
warnings.filterwarnings("ignore", category=FutureWarning, message="Arguments other than a weight enum or `None`.*?")
warnings.filterwarnings("ignore", category=UserWarning, message="Arguments other than a weight enum or `None`.*?")
warnings.filterwarnings("ignore", category=UserWarning, message="torch.nn.utils.weight_norm is deprecated.*?")

__version__ = "2.0.1"

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
    clip: vs.VideoNode, model: int = 0, render_factor: int = 24, sat: list = [1.0,1.0], hue: list = [0.0,0.0], dd_model: int = 1, dd_render_factor: int = 24, dd_tweak_luma_bind: list = [False, 0.4, 0.4], dd_tweak_bright: float = 0.0, dd_tweak_cont: float = 1, dd_tweak_gamma: float = 1.0, dd_method: int = 2, merge_weight: float = 0.5, dd_method_params: list = [0.6, 2.0, 0.15, 0.2, False], chroma_resize: bool = True, luma_mask: list = [0.3, 0.6, 1.0], dark_darkness: list = [False, 0.1, 0.2, 0.5, 0.8], color_stabilizer: list = [False, False, False, 5, 'arithmetic', True], color_limiter: list = [False, 0.02], device_index: int = 0, n_threads: int = 8, dd_num_streams: int = 1, torch_hub_dir: str = model_dir, stack_frames: bool = True) -> vs.VideoNode:
    """A Deep Learning based project for colorizing and restoring old images and video 

    :param clip:               clip to process, only RGB24 format is supported.
    :param model:              deoldify model to use (default = 0):
                                  0 = ColorizeVideo_gen
                                  1 = ColorizeStable_gen
                                  2 = ColorizeArtistic_gen
    :param render_factor:      render factor for the model, range: 10-44 (default = 24).
    :param sat:                list with the saturation parameters to apply to color models (default = [1,1])
                                  [0] : saturation for deoldify, range [0-10] 
                                  [1] : saturation for ddcolor, range [0-10]                                  
    :param hue:                list with the hue parameters to apply to color models (default = [0,0])    
                                  [0] : hue for deoldify
                                  [1] : hue for ddcolor                    
    :param dd_model:           ddcolor model (default = 1): 
                                  0 = ddcolor_modelscope, 
                                  1 = ddcolor_artistic
    :param dd_render_factor:   ddcolor input size equivalent to render_factor, if = 0 will be auto selected 
                                (default = 24) [range: 0, 10-64] 
    :param dd_tweak_luma_bind: parameters for luma constrained ddcolor preprocess
                                  [0] : luma_constrained_tweak -> luma constrained ddcolor preprocess enabled (default = False), range: [True, False]
                                            when enaabled the average luma of a video clip will be forced to don't be below the value
                                            defined by the parameter "luma_min". The function allow to modify the gamma
                                            of the clip if the average luma is below the parameter "gamma_luma_min"      
                                  [1] : luma_min -> luma (%) min value for tweak activation (default = 0, non activation), range [0-1]
                                  [2] : gamma_luma_min -> luma (%) min value for gamma tweak activation (default = 0, non activation), range [0-1]
    :param dd_tweak_bright     ddcolor tweak's bright (default = 0)
    :param dd_tweak_cont       ddcolor tweak's constrast (default = 1)
    :param dd_tweak_gamma      ddcolor tweak's gamma (default = 1)                                  
    :param dd_method:          method used to combine deoldify with ddcolor (default = 2): 
                                  0 : deoldify only (no merge)
                                  1 : ddcolor only (no merge)                                 
                                  2 : Simple Merge: 
                                        the images are combined using a weighted merge, where the parameter clipb_weight
                                        represent the weight assigned to the colors provided by ddcolor() 
                                  3 : Adaptive Luma Merge:
                                        given that the ddcolor() perfomance is quite bad on dark scenes, the images are 
                                        combinaed by decreasing the weight assigned to ddcolor() when the luma is 
                                        below a given threshold given by: luma_threshold. 
                                        For example with: luma_threshold = 0.6 and alpha = 1, the weight assigned to 
                                        ddcolor() will start to decrease linearly when the luma < 60% till "min_weight".
                                        For alpha=2, begins to decrease quadratically.      
                                  4 : Constrained Chroma Merge:
                                        given that the colors provided by deoldify() are more conservative and stable 
                                        than the colors obtained with ddcolor() images are combined by assigning
                                        a limit to the amount of difference in chroma values between deoldify() and
                                        ddcolor() this limit is defined by the parameter threshold. The limit is applied
                                        to the image converted to "YUV". For example when threshold=0.1, the chroma
                                        values "U","V" of ddcolor() image will be constrained to have an absolute
                                        percentage difference respect to "U","V" provided by deoldify() not higher than 10%  
                                  5 : Luma Masked Merge:   
                                        the clips are combined using a mask merge, the pixels of clipb with luma < luma_mask_limit
                                        will be filled with the pixels of clipa, if the parameter clipm_weight > 0
                                        the masked image will be merged with clipa  
    :param ddcolor_weight      weight given to ddcolor clip in all merge methods, range [0-1] (0.01=1%)                                    
    :param dd_method_params:   list with the parameters to apply to selected dd_method:
                                 [0] : luma_threshold, used by: AdaptiveLumaMerge, range [0-1] (0.01=1%) 
                                 [1] : alpha (float), used by: AdaptiveLumaMerge, range [>0] 
                                 [2] : min_weight, used by: AdaptiveLumaMerge, range [0-1] (0.01=1%) 
                                 [3] : chroma_threshold (%), used by: ConstrainedChromaMerge range [0-1] (0.01=1%) 
                                 [4] : if true invert the clip order in the Merge methods
    :param chroma_resize:      if True will be enabled the chroma_resize: the cololorization will be applied to a clip with the same 
                               size used for the models inference(), but the final resolution will be the one of the original clip.
    :param luma_mask:          parameters for method: Luma Masked Merge (dd_method=5)
                                 [0] : luma_mask_limit, luma limit for build the mask used in Luma Masked Merge, range [0-1] (0.01=1%) 
                                 [1] : luma_white_limit, if > luma_mask_limit will be applied a gradient till luma_white_limit, range [0-1] (0.01=1%)
                                 [2] : luma_mask_sat: if < 1 the ddcolor dark pixels will substitute with the desaturated deoldify pixels, range [0-1] (0.01=1%) 
    :param dark_darkness:      parameters for dark the portion of the clip with luma below a given threshold
                                 [0] : darkness_enabled (bool), if true the filter is enabled 
                                 [1] : dark_threshold, luma threshold to select the dark area, range [0-1] (0.01=1%)  
                                 [2] : white_threshold, if > dark_threshold will be applied a gradient till white_threshold, range [0-1] (0.01=1%)  
                                 [3] : dark_sat: amount of desaturation to apply to the dark area, range [0-1] 
                                 [4] : dark_bright (float): darkness parameter it used to reduce the "V" component in "HSV" colorspace, range [0, 1] 
    :param color_stabilizer:   parameters for the temporal color stabilizer
                                 [0] : colstab_merge_enabled (bool), if true the filter will be applied after the merge of Deoldify and DDColor
                                 [1] : colstab_deoldify_enabled (bool), if true the filter will be applied after Deoldify
                                 [2] : colstab_ddcolor_enabled (bool), if true the filter will be applied after DDColor
                                 [3] : colstab_nframes, number of frames to be used in the stabilizer, range[3-15]
                                 [4] : colstab_mode, type of average used by the stabilizer: range['arithmetic', 'weighted']                                 
                                 [5] : colstab_scenechange, if true the futures frames will not be used in case of scene change detection, range [True,False] 
    :param color_limiter:      parameters for the temporal color limiter
                                     [0] : colimit_enabled (bool), if true the filter will be applied after the merge of Deoldify and DDColor
                                     [1] : colimit_deviation, chroma of current frame will be forced to be inside the range defined by "deviation", range[0.01-0.5]
    :param device_index:       device ordinal of the GPU, choices: GPU0...GPU7, CPU=99 (default = 0)
    :param n_threads:          number of threads used by numpy, range: 1-32 (default = 8)
    :param dd_num_streams:     number of CUDA streams to enqueue the kernels (default = 1)
    :param torch_hub_dir:      torch hub dir location, default is model directory,
                               if set to None will switch to torch cache dir.
    """
    
    if (not torch.cuda.is_available() and device_index != 99):
        raise vs.Error("ddeoldify: CUDA is not available")

    if not isinstance(clip, vs.VideoNode):
        raise vs.Error("ddeoldify: this is not a clip")

    if model not in range(3):
        raise vs.Error("ddeoldify: model must be 0,1,2")

    if os.path.getsize(os.path.join(model_dir, "ColorizeVideo_gen.pth")) == 0:
        raise vs.Error("ddeoldify: model files have not been downloaded.")

    if device_index > 7 and device_index != 99:
        raise vs.Error("ddeoldify: wrong device_index, choices are: GPU0...GPU7, CPU=99")
    
    if render_factor not in range(10, 45):
        raise vs.Error("ddeoldify: dd_render_factor must be between: 10-44")
    
    if dd_render_factor != 0 and dd_render_factor not in range(10, 65):
        raise vs.Error("ddeoldify: dd_render_factor must be between: 10-64")
            
    if n_threads not in range(1, 32):
        n_threads = 8

    if clip.format.id != vs.RGB24:
        # clip not in RGB24 format, it will be converted
        if (clip.format.color_family == "YUV"):
            clip = clip.resize.Bicubic(format=vs.RGB24, matrix_in_s="709", range_s="limited", dither_type="error_diffusion") 
        else:
            clip = clip.resize.Bicubic(format=vs.RGB24, range_s="limited") 
       
    os.environ['NUMEXPR_MAX_THREADS'] = str(n_threads)
     
    #choices: GPU0...GPU7, CPU=99 
    device.set(device=DeviceId(device_index))
    
    if torch_hub_dir != None:
        torch.hub.set_dir(torch_hub_dir)      
         
    if dd_render_factor == 0:
        dd_render_factor = min(max(math.trunc(0.4 * clip.width / 16), 16), 48)
    
    # frame size calculation for inference()    
    dd_frame_size = min(dd_render_factor * 16, clip.width)
    
    clipb_weight = merge_weight
    
    if chroma_resize:
        clip_orig = clip;
        clip = clip.resize.Spline64(width=dd_frame_size, height=dd_frame_size) 
                
    # unpack method params
    luma_threshold = dd_method_params[0]
    alpha = dd_method_params[1]
    min_weight = dd_method_params[2]
    chroma_threshold = dd_method_params[3]
    invert_clips = dd_method_params[4]

     # unpack luma_mask
    luma_mask_limit = luma_mask[0]
    luma_white_limit = luma_mask[1]
    luma_mask_sat = luma_mask[2]  

    # unpack dark_darkness
    darkness_enabled = dark_darkness[0]
    dark_threshold = dark_darkness[1]
    white_threshold = dark_darkness[2]
    dark_sat = dark_darkness[3]
    dark_bright = -dark_darkness[4] #change the sign to reduce the bright    
    
    # unpack color_stabilizer
    colstab_merge_enabled = color_stabilizer[0]
    colstab_deoldify_enabled = color_stabilizer[1]
    colstab_ddcolor_enabled = color_stabilizer[2]
    colstab_nframes = color_stabilizer[3]
    colstab_mode = color_stabilizer[4]
    colstab_scenechange = color_stabilizer[5]
    
     # unpack color_limiter
    colimit_enabled = color_limiter[0]
    colimit_deviation = color_limiter[1]
    
    if dd_method == 0 or (dd_method != 1):
        match model:
            case 0:
                colorizer = get_image_colorizer(root_folder=Path(package_dir), artistic=False,isvideo=True) 
            case 1:
                colorizer = get_image_colorizer(root_folder=Path(package_dir), artistic=False,isvideo=False) 
            case 2:
                colorizer = get_image_colorizer(root_folder=Path(package_dir), artistic=True,isvideo=False) 
            
        clipa = _get_deoldify_colorize(clip, colorizer=colorizer, render_factor=render_factor)        
    else:
        clipa = None
    
    if dd_method == 1 or (dd_method != 0):                
        clipb = _get_ddcolor_colorize(clip,  model=dd_model, input_size=dd_frame_size, tweak_luma_bind=dd_tweak_luma_bind, bright=dd_tweak_bright, cont=dd_tweak_cont, gamma=dd_tweak_gamma, device_index=device_index, num_streams=dd_num_streams)
    else:
        clipb = None
    
    if (clipa is not None) and colstab_deoldify_enabled:
            clipa = vs_clip_color_stabilizer(clipa, nframes = colstab_nframes, mode=colstab_mode, scenechange = colstab_scenechange)

    if (clipb is not None) and colstab_ddcolor_enabled:
            clipb = vs_clip_color_stabilizer(clipb, nframes = colstab_nframes, mode=colstab_mode, scenechange = colstab_scenechange)
    
    if invert_clips:     
        clip_colored = combine_models(clipa=clipb, clipb=clipa, sat=sat, hue=hue, method=dd_method, clipb_weight=clipb_weight, luma_threshold=luma_threshold, alpha=alpha, min_weight=min_weight, chroma_threshold=chroma_threshold, luma_mask_limit=luma_mask_limit, luma_white_limit=luma_white_limit, luma_mask_sat=luma_mask_sat)
    else:
        clip_colored = combine_models(clipa=clipa, clipb=clipb, sat=sat, hue=hue, method=dd_method, clipb_weight=clipb_weight, luma_threshold=luma_threshold, alpha=alpha, min_weight=min_weight, chroma_threshold=chroma_threshold, luma_mask_limit=luma_mask_limit, luma_white_limit=luma_white_limit, luma_mask_sat=luma_mask_sat)

    if darkness_enabled:
        clip_colored = darkness_tweak(clip_colored, dark_threshold=dark_threshold, white_threshold=white_threshold, dark_sat=dark_sat, dark_bright=dark_bright)  
    
    if colimit_enabled:
        clip_colored = vs_clip_chroma_stabilizer(clip_colored, deviation = colimit_deviation)    
    
    if colstab_merge_enabled:
        clip_colored = vs_clip_color_stabilizer(clip_colored, nframes = colstab_nframes, mode=colstab_mode, scenechange = colstab_scenechange)
            
    if chroma_resize:
        clip_colored = clip_colored.resize.Spline64(width=clip_orig.width, height=clip_orig.height) 
        return recover_clip_luma(clip_orig, clip_colored)
    else:
        return clip_colored

"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
Chroma resize filter, derived from ddeoldify 
"""
def dd_chroma_resize(clip_hires: vs.VideoNode, clip_lowres: vs.VideoNode) -> vs.VideoNode:
    clip_resized = clip_lowres.resize.Spline64(width=clip_hires.width, height=clip_hires.height) 
    return recover_clip_luma(clip_hires, clip_resized)

"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
Video color stabilization filter, derived from ddeoldify
"""
def dd_video_stabilizer(clip: vs.VideoNode, chroma_resize: list = [False, 32], color_smoothing: list = [False, 0.1, 0.2, 0.5, 0.8], color_stabilizer: list = [False, 5, 'arithmetic', 1, True], color_limiter: list = [False, 0.02]) -> vs.VideoNode:
    """Video color stabilization filter, derived from ddeoldify. 
        :param clip:               clip to process, only RGB24 format is supported.
        :param chroma_resize:      parameters for the chroma resizer pre-filter 
                                     [0] : if True the chroma resizer is enabled 
                                     [1] : render_factor to apply to the chroma_resizer
                                   size used for the models inference(), but the final resolution will be the one of the original clip.
        :param color_smoothing:    parameters for dark the portion of the clip with luma below a given threshold
                                     [0] : darkness_enabled (bool), if true the filter is enabled 
                                     [1] : dark_threshold, luma threshold to select the dark area, range [0-1] (0.01=1%)  
                                     [2] : white_threshold, if > dark_threshold will be applied a gradient till white_threshold, range [0-1] (0.01=1%)  
                                     [3] : dark_sat: amount of desaturation to apply to the dark area, range [0-1] 
                                     [4] : dark_bright (float): darkness parameter it used to reduce the "V" component in "HSV" colorspace, range [0, 1] 
        :param color_stabilizer:   parameters for the temporal color stabilizer
                                     [0] : colstab_merge_enabled (bool), if true the filter will be applied after the merge of Deoldify and DDColor
                                     [1] : colstab_nframes, number of frames to be used in the stabilizer, range[3-31]
                                     [2] : colstab_mode, type of average used by the stabilizer: range['arithmetic', 'weighted']
                                     [3] : colstab_steps: number of average repetition (max 3) 
                                     [4] : colstab_scenechange, if true the futures frames will not be used in case of scene change detection, range [True,False] 
        :param color_limiter:      parameters for the temporal color limiter
                                     [0] : colimit_enabled (bool), if true the filter will be applied after the merge of Deoldify and DDColor
                                     [1] : colimit_deviation, chroma of current frame will be forced to be inside the range defined by "deviation", range[0.01-0.5]
        
    """

    if not isinstance(clip, vs.VideoNode):
        raise vs.Error("ddeoldify_video_stabilizer: this is not a clip")

    if clip.format.id != vs.RGB24:
        # clip not in RGB24 format, it will be converted
        if (clip.format.color_family == "YUV"):
            clip = clip.resize.Bicubic(format=vs.RGB24, matrix_in_s="709", range_s="limited", dither_type="error_diffusion") 
        else:
            clip = clip.resize.Bicubic(format=vs.RGB24, range_s="limited") 
        
    # unpack chroma_resize
    chroma_resize_enabled = chroma_resize[0]
    render_factor = chroma_resize[1]
    
    if render_factor != 0 and render_factor not in range(16, 65):
        raise vs.Error("ddeoldify_video_stabilizer: dd_render_factor must be between: 16-65")
        
    if render_factor == 0:
        render_factor = min(max(math.trunc(0.4 * clip.width / 16), 16), 64)
    
    # frame size calculation for inference()    
    frame_size = min(render_factor * 16, clip.width)
    
    if chroma_resize_enabled:
        clip_orig = clip;
        clip = clip.resize.Spline64(width=frame_size, height=frame_size) 
            
    # unpack color_smoothing
    darkness_enabled = color_smoothing[0]
    dark_threshold = color_smoothing[1]
    white_threshold = color_smoothing[2]
    dark_sat = color_smoothing[3]
    dark_bright = -color_smoothing[4] #change the sign to reduce the bright
    
    # unpack color_stabilizer
    colstab_merge_enabled = color_stabilizer[0]
    colstab_nframes = color_stabilizer[1]
    colstab_mode = color_stabilizer[2]
    colstab_steps = color_stabilizer[3]
    colstab_scenechange = color_stabilizer[4]
    
    # unpack color_limiter
    colimit_enabled = color_limiter[0]
    colimit_deviation = color_limiter[1]
    
    if colstab_steps not in range(1,4):
        raise vs.Error("ddeoldify: steps must be between: 1-3")
                
    clip_colored = clip
    
    if darkness_enabled:
        clip_colored = darkness_tweak(clip_colored, dark_threshold=dark_threshold, white_threshold=white_threshold, dark_sat=dark_sat, dark_bright=dark_bright) 
    
    if colimit_enabled:
        clip_colored = vs_clip_chroma_stabilizer(clip_colored, deviation=colimit_deviation)
        
    if colstab_merge_enabled:
        clip_colored = _color_stabilizer_ex(clip_colored, nframes = colstab_nframes, mode=colstab_mode, steps=colstab_steps, scenechange = colstab_scenechange)
        
    if chroma_resize_enabled:
        clip_colored = clip_colored.resize.Spline64(width=clip_orig.width, height=clip_orig.height) 
        return recover_clip_luma(clip_orig, clip_colored)
    else:
        return clip_colored

"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
wrapper to function vs_clip_color_stabilizer() with management parameter "steps"
"""
def _color_stabilizer_ex(clip: vs.VideoNode = None, nframes: int = 5, mode: str = "center", steps: int = 1, scenechange: bool = True) -> vs.VideoNode:
    
    max_steps = max(min(steps, 3), 1)
    for i in range(0, max_steps):
        clip = vs_clip_color_stabilizer(clip, nframes = nframes, mode=mode, scenechange = scenechange)
            
    return clip

"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
wrapper to deoldify 
"""
def _get_deoldify_colorize(clip: vs.VideoNode, colorizer: ModelImageVisualizer = None, render_factor: int = 24) -> vs.VideoNode: 
    
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
wrapper to function ddcolor() with tweak pre-process and recover luma post-process
"""
def _get_ddcolor_colorize(clip: vs.VideoNode, model: int = 0, input_size: int = 384, tweak_luma_bind: list = [False, 0.0, 0.0], bright: float = 0, cont: float = 1, gamma: float = 1, luma_stab: float = 0, device_index: int = 0, num_streams: int = 1) -> vs.VideoNode:
    #if luma_stab > 0:
    #    clipb = clip_luma_stabilizer(clip, luma_stab)
    # unpack tweak_luma_bind params
    luma_constrained_tweak=tweak_luma_bind[0]
    luma_min = tweak_luma_bind[1] 
    gamma_luma_min = tweak_luma_bind[2]
    if (luma_constrained_tweak):
        clipb = tweak(clip, cont=cont) # constrained tweak on image contrast is not implemented
        clipb = constrained_tweak(clipb, luma_min = luma_min, gamma=gamma, gamma_luma_min = gamma_luma_min)
    else:
        clipb = tweak(clip, bright=bright, cont=cont, gamma=gamma)
    # adjusting clip's color space to RGBH for vsDDColor
    clipb = ddcolor(clipb.resize.Bicubic(format=vs.RGBH, range_s="limited"), model=model, input_size=input_size, device_index=device_index, num_streams=num_streams)    
    # adjusting color space to RGB24 for deoldify
    clipb_rgb = clipb.resize.Bicubic(format=vs.RGB24, range_s="limited")
    if (bright == 0 and cont == 1 and gamma == 1) or (luma_constrained_tweak == False):
        return clipb_rgb
    else:
        return recover_clip_luma(clip, clipb_rgb)

