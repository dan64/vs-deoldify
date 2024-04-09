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
Library of Vapoursynth utility functions.
"""

import vapoursynth as vs
import math
import numpy as np
import cv2
from PIL import Image


"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
function to convert a VideoFrame in Pillow image 
(why not available in Vapoursynth ?) 
"""     
def frame_to_image(frame: vs.VideoFrame) -> Image:
    npArray = np.dstack([np.asarray(frame[plane]) for plane in range(frame.format.num_planes)])
    return Image.fromarray(npArray, 'RGB')

"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
function to convert a VideoFrame in Pillow image 
(why not available in Vapoursynth ?) 
"""     
def frame_to_np_array(frame: vs.VideoFrame) -> np.ndarray:
    npArray = np.dstack([np.asarray(frame[plane]) for plane in range(frame.format.num_planes)])
    return npArray
    
"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
function to convert a Pillow image in VideoFrame 
(why not available in Vapoursynth ?) 
"""
def image_to_frame(img: Image, frame: vs.VideoFrame) -> vs.VideoFrame:
    npArray = np.array(img)
    [np.copyto(np.asarray(frame[plane]), npArray[:, :, plane]) for plane in range(frame.format.num_planes)]
    return frame

"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
function to convert a np.array() image in VideoFrame 
"""
def np_array_to_frame(npArray: np.ndarray, frame: vs.VideoFrame) -> vs.VideoFrame:
    [np.copyto(np.asarray(frame[plane]), npArray[:, :, plane]) for plane in range(frame.format.num_planes)]
    return frame
