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
Library of functions used by "ddeoldify" to restore color and change the hue of frames.
"""
import vapoursynth as vs
import math
import numpy as np
import cv2
from PIL import Image

from .nputils import *

"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
Restore the colors of past/future frame. The restore is applied using a mask
to selectect only the gray images on HSV color space.
The ranges that OpenCV manage for HSV format are the following:
- Hue range is [0,179], 
- Saturation range is [0,255] 
- Value range is [0,255].
For the 8-bit images, H is converted to H/2 to fit to the [0,255] range. 
So the range of hue in the HSV color space of OpenCV is [0,179]
"""
def restore_color(img_color: Image = None, img_gray: Image = None, sat: float=1.0, tht: int = 15, weight: float = 0, tht_scen: float=0.15, hue_adjust: str='none', return_mask: bool=False) -> Image:
      
    np_color = np.asarray(img_color)
    np_gray = np.asarray(img_gray)
       
    hsv_color = cv2.cvtColor(np_color, cv2.COLOR_RGB2HSV)
    hsv_gray = cv2.cvtColor(np_gray, cv2.COLOR_RGB2HSV)
    
    # desatured the color image
    hsv_color[:, :, 1] = hsv_color[:, :, 1] * min(max(sat, 0), 1)
    
    np_color_sat = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2RGB)
    
    hsv_s = hsv_gray[:, :, 1]
    
    hsv_mask = np.where(hsv_s < tht, 255, 0) # white only gray pixels
    
    scenechange = np.mean(hsv_mask)/255
    
    if (tht_scen > 0 and tht_scen < 1 and  scenechange > tht_scen):
        if hue_adjust:
            return adjust_hue_range(img_gray, hue_adjust=hue_adjust)
        else:
            return img_gray
    
    mask_rgb = np_gray.copy();
    
    for i in range(3):
        mask_rgb[:,:,i] = hsv_mask
    
    if return_mask:
        return Image.fromarray(mask_rgb,'RGB').convert('RGB') 
        
    np_restored = np_image_mask_merge(np_gray, np_color_sat, mask_rgb)
    
    if weight > 0:
        np_restored = np_weighted_merge(np_restored, np_gray, weight)
    
    img_restored = Image.fromarray(np_restored,'RGB').convert('RGB') 
    
    if hue_adjust:
        return adjust_hue_range(img_restored, hue_adjust=hue_adjust)
    else:
        return img_restored

"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
Change a given range of colors in HSV color space. 
The range is defined by the hue values in degree (range: 0-360)
In OpenCV, for the 8-bit images, H is converted to H/2 to fit to the [0,255] range. 
So the range of hue in the HSV color space of OpenCV is [0,179]
"""
def adjust_hue_range(img_color: Image = None, hue_adjust: str='none', return_mask: bool=False) -> Image:
    
    if hue_adjust=='none' or hue_adjust=='':
        return img_color
    
    param = _parse_hue_adjust(hue_adjust)
    
    hue_range = param[0]
    sat = param[1]
    weight = param[2]
            
    return adjust_chroma(img_color=img_color, hue_range=hue_range, sat=sat, weight=weight, return_mask=return_mask)

def adjust_chroma(img_color: Image = None, hue_range: str='none', sat: float = 0.3, weight: float = 0, return_mask: bool=False) -> Image:
    
    if hue_range=='none' or hue_range=='':
        return img_color
        
    np_color = np.asarray(img_color)
    
    np_gray = np_color.copy()    
    np_gray = cv2.cvtColor(np_gray, cv2.COLOR_RGB2HSV)
    
    hsv_color = cv2.cvtColor(np_color, cv2.COLOR_RGB2HSV)       
    
    # desatured the color image to gray
    np_gray[:, :, 1] = np_gray[:, :, 1] * min(max(sat, 0), 1)
    
    np_gray_rgb = cv2.cvtColor(np_gray, cv2.COLOR_HSV2RGB)
    
    hsv_s = hsv_color[:, :, 0]
    
    cond = _build_hue_conditions(hsv_s, hue_range)  
    
    hsv_mask = np.where(cond, 255, 0) # white only gray pixels
       
    mask_rgb = np_color.copy();
    
    for i in range(3):
        mask_rgb[:,:,i] = hsv_mask
    
    if return_mask:
        return Image.fromarray(mask_rgb,'RGB').convert('RGB') 
        
    np_restored = np_image_mask_merge(np_color, np_gray_rgb, mask_rgb)
    
    if weight > 0:
        np_restored = np_weighted_merge(np_restored, np_gray_rgb, weight)
    
    return Image.fromarray(np_restored,'RGB').convert('RGB') 

def np_adjust_chroma2(np_color_rgb: np.ndarray, np_gray_rgb: np.ndarray, hue_range: str='none', return_mask: bool=False) -> np.ndarray:
    
    if hue_range=='none' or hue_range=='':
        return np_color_rgb
    
    hsv_color = cv2.cvtColor(np_color_rgb, cv2.COLOR_RGB2HSV)         
    hsv_s = hsv_color[:, :, 0]
    
    cond = _build_hue_conditions(hsv_s, hue_range)  
    
    hsv_mask = np.where(cond, 255, 0) # white only gray pixels
       
    mask_rgb = np_color_rgb.copy();
    
    for i in range(3):
        mask_rgb[:,:,i] = hsv_mask
    
    if return_mask:
        return Image.fromarray(mask_rgb,'RGB').convert('RGB') 
        
    np_restored = np_image_mask_merge(np_color_rgb, np_gray_rgb, mask_rgb)
            
    return np_restored

def _parse_hue_adjust(hue_adjust: str='none') -> ():
    p = hue_adjust.split("|")
    
    if len(p)!=2:
        return None
    
    sw = p[1].split(",") 
    
    return (p[0], float(sw[0]), float(sw[1]))
  

def _build_hue_conditions(hsv_s: np.ndarray=None, hue_range: str= None) -> np.ndarray:
    
    h_range = hue_range.split(",")
    h_len = len(h_range)

    hue_min, hue_max =  _parse_hue_range(h_range[0])  
    # For the 8-bit images, H is converted to H/2 to fit to the [0,255] range.
    c1 = hsv_s > hue_min*0.5
    c2 = hsv_s < hue_max*0.5
    cond = (c1 & c2)
    
    for i in range(1,h_len):
        hue_min, hue_max =  _parse_hue_range(h_range[i])        
        c1 = hsv_s > hue_min*0.5
        c2 = hsv_s < hue_max*0.5
        cond |= (c1 & c2)
    
    return cond

def _parse_hue_range(hue_range: str = None) -> ():
    #For color increments, each block in a given "hue_range" represents a Hue change of 30.     
    match hue_range:
        case "red":
            rng = (0,30) 
        case "orange":
            rng = (30,60)
        case "yellow":
            rng = (60,90) 
        case "yellow-green":
            rng = (90,120)
        case "green":
            rng = (120,150) 
        case "blue-green":
            rng = (150,180)
        case "cyan":
            rng = (180,210) 
        case "blue":
            rng = (210,240)
        case "blue-violet":
            rng = (240,270) 
        case "violet":
            rng = (270,300)
        case "red-violet":
            rng = (300,330) 
        case "rose":
            rng = (330,360)
        case _:
            p = hue_range.split(":")
            if (len(p)==2 and p[0].isnumeric() and p[1].isnumeric()):
                rng = (float(p[0]), float(p[1]))
            else:
                raise vs.Error("ddeoldify: unknown hue name: " + hue_range)    
    
    return rng
    