from __future__ import annotations

import os

os.environ["CUDA_MODULE_LOADING"] = 'LAZY'
os.environ['NUMEXPR_MAX_THREADS'] = '8'

import numpy as np
import math 
from .deoldify import device
from .deoldify.device_id import DeviceId
from PIL import Image

from .deoldify.visualize import *
from .deoldify.adjust import *
from vsddcolor import ddcolor

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*?Your .*? set is empty.*?")
warnings.filterwarnings("ignore", category=UserWarning, message="The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.")
warnings.filterwarnings("ignore", category=FutureWarning, message="Arguments other than a weight enum or `None`.*?")
warnings.filterwarnings("ignore", category=UserWarning, message="Arguments other than a weight enum or `None`.*?")
warnings.filterwarnings("ignore", category=UserWarning, message="torch.nn.utils.weight_norm is deprecated.*?")

__version__ = "1.1.2"


package_dir = os.path.dirname(os.path.realpath(__file__))
model_dir = os.path.join(package_dir, "models")

#configuring torch
torch.backends.cudnn.benchmark=True


import vapoursynth as vs

def ddeoldify(
    clip: vs.VideoNode, model: int = 0, render_factor: int = 21, sat: list = [1,1], hue: list = [0,0], dd_method: int = 0, dd_weight: float = 0.0,
    dd_strength: int = 0, dd_model: int = 0, device_index: int = 0, n_threads: int = 8, dd_num_streams: int = 1, torch_hub_dir: str = model_dir
) -> vs.VideoNode:
    """A Deep Learning based project for colorizing and restoring old images and video 

    :param clip:           clip to process, only RGB24 format is supported.
    :param model:          deoldify model to use (default = 0):
                              0 = ColorizeVideo_gen
                              1 = ColorizeStable_gen
                              2 = ColorizeArtistic_gen
    :param render_factor:  render factor for the model, range: 10-40 (default = 21).
    :param sat:            list with the saturation parameters to apply to color models (default = [1,1])
    :param hue:            list with the hue parameters to apply to color models (default = [0,0])
    :param dd_weight:      weight assigned to ddcolor (default = 0) [range: 0-1], 
                                if = 0 ddcolor will be disabled      
                                if = 1 deoldify will be disabled
    :param dd_method:      method used to combine deoldify with ddcolor (default = 0): 
                              0 : Simple Merge
    :param dd_strength:    ddcolor input size, if = 0 will be auto selected (default = 0) [range: 0-9] 
    :param dd_model:       ddcolor model (default = 0): 
                              0 = ddcolor_modelscope, 
                              1 = ddcolor_artistic
    :param device_index:   device ordinal of the GPU, choices: GPU0...GPU7, CPU=99 (default = 0)
    :param n_threads:      number of threads used by numpy, range: 1-32 (default = 8)
    :param dd_num_streams: number of CUDA streams to enqueue the kernels (default = 1)
    :param torch_hub_dir:  torch hub dir location, default is model directory,
                           if set to None will switch to torch cache dir.
    """
    
    if (not torch.cuda.is_available() and device_index != 99):
        raise vs.Error("deoldify: CUDA is not available")

    if not isinstance(clip, vs.VideoNode):
        raise vs.Error("deoldify: this is not a clip")

    if clip.format.id != vs.RGB24:
        raise vs.Error("deoldify: only RGB24 format is supported")

    if model not in range(3):
        raise vs.Error("deoldify: model must be 0,1,2")

    if os.path.getsize(os.path.join(model_dir, "ColorizeVideo_gen.pth")) == 0:
        raise vs.Error("deoldify: model files have not been downloaded.")

    if device_index > 7 and device_index != 99:
        raise vs.Error("deoldify: wrong device_index, choices are: GPU0...GPU7, CPU=99")
    
    if dd_strength not in range(0, 10):
        raise vs.Error("deoldify: dd_strength must between: 0-9")
            
    if n_threads not in range(1, 32):
        n_threads = 8
    
    # input_size calculation for ddcolor    
    if dd_strength > 0:
        dd_insize = dd_strength * 128
    else:
        dd_insize = min(max(math.trunc(clip.width / 128), 1), 9) * 128

    os.environ['NUMEXPR_MAX_THREADS'] = str(n_threads)
     
    #choices: GPU0...GPU7, CPU=99 
    device.set(device=DeviceId(device_index))
    
    if torch_hub_dir != None:
        torch.hub.set_dir(torch_hub_dir)
    
    match model:
        case 0:
            colorizer = get_image_colorizer(root_folder=Path(package_dir), artistic=False,isvideo=True) 
        case 1:
            colorizer = get_image_colorizer(root_folder=Path(package_dir), artistic=False,isvideo=False) 
        case 2:
            colorizer = get_image_colorizer(root_folder=Path(package_dir), artistic=True,isvideo=False) 
   
    def ddeoldify_colorize(n: int, f: vs.VideoFrame) -> vs.VideoFrame:
        img_orig = frame_to_image(f)
        img_color = colorizer.get_transformed_pil_image(img_orig, render_factor=render_factor, post_process=True)
        f_out = image_to_frame(img_color, f.copy()) 
        return f_out      
    
    if dd_weight < 1.0:        
        clipa = clip.std.ModifyFrame(clip, ddeoldify_colorize) 
    else:
        clipa = None
    
    if dd_weight > 0:    
        clipb = get_ddcolor_colorize(clip,  model=dd_model, input_size=dd_insize, device_index=device_index, num_streams=dd_num_streams)
    else:
        clipb = None
            
    return combine_models(clipa, clipb, sat, hue, dd_method, dd_weight)

def get_ddcolor_colorize(clip: vs.VideoNode, model: int = 0, input_size: int = 512, device_index: int = 0, num_streams: int = 1) -> vs.VideoNode:
    # adjusting clip's color space to RGBH for vsDDColor 
    clipb = ddcolor(clip.resize.Bicubic(format=vs.RGBH, range_s="limited"), model=model, input_size=input_size, device_index=device_index, num_streams=num_streams)    
    # adjusting color space to RGB24 for deoldify
    return clipb.resize.Bicubic(format=vs.RGB24, range_s="limited")

def frame_to_image(frame: vs.VideoFrame) -> Image:
    npArray = np.dstack([np.asarray(frame[plane]) for plane in range(frame.format.num_planes)])
    return Image.fromarray(npArray, 'RGB')


def image_to_frame(img: Image, frame: vs.VideoFrame) -> vs.VideoFrame:
    npArray = np.array(img)
    [np.copyto(np.asarray(frame[plane]), npArray[:, :, plane]) for plane in range(frame.format.num_planes)]
    return frame
