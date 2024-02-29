from __future__ import annotations

import os

import numpy as np
from .deoldify import device
from .deoldify.device_id import DeviceId
from PIL import Image

from .deoldify.visualize import *

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*?Your .*? set is empty.*?")
warnings.filterwarnings("ignore", category=UserWarning, message="The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.")
warnings.filterwarnings("ignore", category=FutureWarning, message="Arguments other than a weight enum or `None`.*?")

__version__ = "1.0.0"

os.environ["CUDA_MODULE_LOADING"] = "LAZY"

package_dir = os.path.dirname(os.path.realpath(__file__))
model_dir = os.path.join(package_dir, "models")

import vapoursynth as vs
core = vs.core

def ddeoldify(
    clip: vs.VideoNode, model: int = 0, render_factor: int = 21, device_index: int = 0, post_process=True   
) -> vs.VideoNode:
    """A Deep Learning based project for colorizing and restoring old images and video 

    :param clip:           clip to process, only RGB24 format is supported.
    :param model:          model to use.
                              0 = ColorizeVideo_gen
                              1 = ColorizeStable_gen
                              2 = ColorizeArtistic_gen
    :param render_factor:  render factor for the model (range: 10-40).
    :param device_index:   device ordinal of the GPU, choices: GPU0...GPU7, CPU=99
    :param post_process:   post_process takes advantage of the fact that human eyes are less sensitive to imperfections in chrominance to save memory.
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

    #choices: GPU0...GPU7, CPU=99 
    device.set(device=DeviceId(device_index))
    
    torch.backends.cudnn.benchmark=True
    torch.hub.set_dir(model_dir)
    
    match model:
        case 0:
            colorizer = get_image_colorizer(root_folder=Path(package_dir), artistic=False,isvideo=True) 
        case 1:
            colorizer = get_image_colorizer(root_folder=Path(package_dir), artistic=False,isvideo=False) 
        case 2:
            colorizer = get_image_colorizer(root_folder=Path(package_dir), artistic=True,isvideo=False) 
   
    def ddeoldify_colorize(n: int, f: vs.VideoFrame) -> vs.VideoFrame:
        img_orig = frame_to_image(f)
        img_color = colorizer.get_transformed_pil_image(img_orig, render_factor=render_factor, post_process=post_process)
        f_out = image_to_frame(img_color, f.copy()) 
        return f_out
    
    return core.std.ModifyFrame(clip, clip, ddeoldify_colorize) 

def frame_to_image(frame: vs.VideoFrame) -> Image:
    npArray = np.dstack([np.asarray(frame[plane]) for plane in range(frame.format.num_planes)])
    return Image.fromarray(npArray, 'RGB')


def image_to_frame(img: Image, frame: vs.VideoFrame) -> vs.VideoFrame:
    npArray = np.array(img)
    [np.copyto(np.asarray(frame[plane]), npArray[:, :, plane]) for plane in range(frame.format.num_planes)]
    return frame
