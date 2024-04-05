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
Library of functions used by "ddeoldify" to improve the finl output of colored 
images and clips obtained with "deoldify" and "ddcolor".
"""
import vapoursynth as vs
import math
import numpy as np
import cv2
from PIL import Image, ImageMath
from functools import partial

"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
main function used to combine the colored images with deoldify() and ddcolor()
"""
def combine_models(clip_a: vs.VideoNode = None, clip_b: vs.VideoNode = None, method: int = 0, sat: list = [1,1], hue: list = [0,0], clipb_weight: float = 0.5, CMC_p: float = 0.2, LMM_p: list = [0.3, 0.6, 1.0], ALM_p: list = [0.3, 0.6, 1.0], invert_clips: bool = False) -> vs.VideoNode:

    #vs.core.log_message(2, "combine_models: method=" + str(method) + ", clipa = " + str(clipa) + ", clipb = " + str(clipb))
    
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
        clipa = tweak(clipa, hue=hue[0], sat=sat[0])
        if clipb is None: return clipa
    
    if clipb is not None:
        clipb = tweak(clipb, hue=hue[1], sat=sat[1])
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
        raise vs.Error("deoldify: only dd_method=(0,5) is supported")
"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
the images are combined using a weighted merge, where the parameter clipb_weight
represent the weight assigned to the colors provided by ddcolor() 
"""
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
def LumaMaskedMerge(clipa: vs.VideoNode = None, clipb: vs.VideoNode = None, luma_mask_limit: float = 0.4, luma_white_limit: float = 0.7, luma_mask_sat = 1.0, clipm_weight: float = 0.5) -> vs.VideoNode:
   
    if luma_mask_sat < 1:
        #vs.core.log_message(2, "LumaMaskedMerge: mask_sat = " + str(luma_mask_sat))   
        clipc = tweak(clipa, sat=luma_mask_sat)
    else:
        clipc = clipa
    def merge_frame(n, f, weight: float = 0.5, luma_limit: float = 0.4, white_limit: float = 0.7):                
        img1 = frame_to_image(f[0])
        img2 = frame_to_image(f[1]) 
        img3 = frame_to_image(f[2]) 
        if luma_limit == white_limit:
            #vs.core.log_message(2, "frame[" + str(n) + "]: luma_limit = " + str(luma_limit)) 
            img_masked = image_luma_merge(img3, img2, luma_limit)
        else:
            img_masked = w_image_luma_merge(img3, img2, luma_limit, white_limit)
        if clipm_weight < 1:
            img_m = image_weighted_merge(img1, img_masked, weight)
        else:
            img_m = img_masked
        return image_to_frame(img_m, f[0].copy())                
    clipm = clipa.std.ModifyFrame(clips=[clipa, clipb, clipc], selector=partial(merge_frame, weight = clipm_weight, luma_limit = luma_mask_limit, white_limit = luma_white_limit))
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
def AdaptiveLumaMerge(clipa: vs.VideoNode = None, clipb: vs.VideoNode = None, luma_threshold: float = 0.6, alpha: float = 1.0, clipb_weight: float = 0.5, min_weight: float = 0.15 ) -> vs.VideoNode:    
    
    def merge_frame(n, f, luma_limit: float = 0.6, min_w: float = 0.15, alpha: float = 1.0, weight: float = 0.5):                
        img1 = frame_to_image(f[0])
        img2 = frame_to_image(f[1]) 
        luma = get_pil_luma(img2)
        if luma < luma_limit:
            bright_scale = pow(luma/luma_limit, alpha)
            w = max(weight * bright_scale, min_w)
        else:
            w = weight
        #vs.core.log_message(2, "Luma(" + str(n) + ") = " + str(luma) + ", weight = " + str(w))        
        img_m = Image.blend(img1, img2, w)        
        return image_to_frame(img_m, f[0].copy())                
    clipm = clipa.std.ModifyFrame(clips=[clipa, clipb], selector=partial(merge_frame, luma_limit = luma_threshold, min_w = min_weight, alpha=alpha, weight = clipb_weight))
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
def ConstrainedChromaMerge(clipa: vs.VideoNode = None, clipb: vs.VideoNode = None, clipb_weight: float = 0.5, chroma_threshold: float = 0.2) -> vs.VideoNode:
    
    def merge_frame(n, f, level: float = 0.2, weight: float = 0.5):    
        img1 = frame_to_image(f[0])
        img2 = frame_to_image(f[1])        
        img_m = chroma_stabilizer(img1, img2, level, weight)
        return image_to_frame(img_m, f[0].copy())                
    clipm = clipa.std.ModifyFrame(clips=[clipa, clipb], selector=partial(merge_frame, level = chroma_threshold, weight=clipb_weight))
    return clipm

"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
The the pixels with luma below dark_threshold will be desaturared to level defined
by the dark_sat parameter.
"""
def vs_chroma_bright_tweak(clip: vs.VideoNode = None, dark_threshold: float = 0.3, white_threshold: float = 0.6, dark_sat: float = 0.8, dark_bright: float = -0.10) -> vs.VideoNode:      
    def merge_frame(n, f, dark_limit: float = 0.3, white_limit: float = 0.6, dark_bright: float = -0.10, dark_sat: float = 0.8):                
        img1 = frame_to_image(f)
        img2 = image_tweak(img1, bright=dark_bright, sat=dark_sat) 
        if dark_limit == white_limit:
            img_m = image_luma_merge(img2, img1, dark_limit)
        else:
            img_m = w_image_luma_merge(img2, img1, dark_limit, white_limit)
        return image_to_frame(img_m, f.copy())                
    return clip.std.ModifyFrame(clips=clip, selector=partial(merge_frame, dark_limit=dark_threshold, white_limit=white_threshold, dark_bright=dark_bright, dark_sat=dark_sat))

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
Description:
------------------------------------------------------------------------------- 
convert an NP image to gray or B&W if threshold > 0 
"""
def np_rgb_to_gray(img_np: np.ndarray, threshold: float = 0) -> np.ndarray:

    R = img_np[:, :, 0]
    G = img_np[:, :, 1]
    B = img_np[:, :, 2]

    R = R * 0.299
    G = G * 0.587
    B = B * 0.114

    tresh = round(threshold*255)
    
    luma_np = R+G+B
    
    gray_np = img_np.copy()
    
    for i in range(3):
       if threshold > 0:
           gray_np[:,:,i] = np.where(luma_np > tresh, 255, 0)
       else:
           gray_np[:,:,i] = luma_np
           
    return gray_np  

def w_np_rgb_to_gray(img_np: np.ndarray, dark_luma: float = 0, luma_white : float = 0.90) -> np.ndarray:

    R = img_np[:, :, 0]
    G = img_np[:, :, 1]
    B = img_np[:, :, 2]

    R = R * 0.299
    G = G * 0.587
    B = B * 0.114

    luma_np = R+G+B
    
    gray_np = img_np.copy()
    gray_np = gray_np.astype(float)
    
    if dark_luma > 0: 
        max_white = round(luma_white*255)

        tresh = min(round(dark_luma*255), max_white-10)
    
        grad = round(1/(max_white - tresh), 3)
           
        luma_grad = ((luma_np - tresh)*grad).astype(float)
        
        weighted_luma = array_min_max(luma_grad, 0.0, 1.0, np.float32)
    
        for i in range(3):
           gray_np[:,:,i] = weighted_luma
    
    else:
        for i in range(3):
            gray_np[:,:,i] = luma_np/255
           
    return gray_np.astype(float)  

"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
convert an image to gray or B&W if threshold > 0 
"""
def rgb_to_gray(img: Image, threshold: float = 0) -> Image:
        
    gray_np = np_rgb_to_gray(np.array(img), threshold)
               
    return Image.fromarray(gray_np, 'RGB')       

"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
merge image1 with image2 using the mask (white->img2, black->img1) 
"""
def np_image_mask_merge(img1_np: np.ndarray, img2_np: np.ndarray, 
                        mask_np: np.ndarray) -> np.ndarray:
    
    mask_white = (mask_np / 255).astype(float) # pass only white
    mask_black = (1 - mask_white).astype(float)    # pass only black

    img_np = img1_np.copy(); 
    
    img_m = img1_np * mask_black + img2_np * mask_white

    for i in range(3):
        img_np[:,:,i] = img_m[:,:,i].clip(0, 255).astype(int)

    return img_np 

def w_np_image_mask_merge(img1_np: np.ndarray, img2_np: np.ndarray, 
                        mask_w_np: np.ndarray) -> np.ndarray:
    
    mask_white = mask_w_np
    mask_black = (1 - mask_white)

    img_np = img1_np.copy(); 
    
    img_m = np.multiply(img1_np, mask_black).clip(0,255).astype(int) + np.multiply(img2_np, mask_white).clip(0, 255).astype(int)

    for i in range(3):
        img_np[:,:,i] = img_m[:,:,i].clip(0, 255).astype(int)

    return img_np 
    
"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
merge image1 with image2 using the image mask (white->img2, black->img1) 
"""
def image_mask_merge(img1: Image, img2: Image, mask: Image) -> Image:
    img1_np = np.array(img1)
    img2_np = np.array(img2)
    mask_np = np.array(mask)
    
    img_np = np_image_mask_merge(img1_np, img2_np, mask_np) 
    
    return Image.fromarray(img_np, 'RGB') 

"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
merge image1 with image2 using the image mask (img2_white->img2, img2_black->img1) 
"""
def image_luma_merge(img1: Image, img2: Image, luma: float = 0) -> Image:
    img1_np = np.array(img1)
    img2_np = np.array(img2)
    # the mask is built using the second image
    mask_np = np_rgb_to_gray(img2_np, luma)
    
    img_np = np_image_mask_merge(img1_np, img2_np, mask_np) 
    
    return Image.fromarray(img_np, 'RGB') 

def w_image_luma_merge(img1: Image, img2: Image, dark_luma: float = 0.3, white_luma = 0.9) -> Image:
    img1_np = np.array(img1)
    img2_np = np.array(img2)
    # the mask is built using the second image
    mask_w_np = w_np_rgb_to_gray(img2_np, dark_luma, white_luma)
    
    img_np = w_np_image_mask_merge(img1_np, img2_np, mask_w_np) 
    
    return Image.fromarray(img_np, 'RGB') 

"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
numpy implementation of image merge on 3 planes, faster than vs.core.std.Merge()
"""
def image_weighted_merge(img1: Image, img2: Image, weight: float = 0.5) -> Image:
    
    img1_np = np.asarray(img1)
    img2_np = np.asarray(img2)
    
    img_new = np.copy(img1_np)

    img_m = np.multiply(img1_np, 1-weight).clip(0, 255).astype(int) + np.multiply(img2_np, weight).clip(0, 255).astype(int)  
    img_new[:, :, 0] = img_m[:, :, 0]
    img_new[:, :, 1] = img_m[:, :, 1]
    img_new[:, :, 2] = img_m[:, :, 2]
            
    return Image.fromarray(img_new)

def np_weighted_merge(img1_np: np.ndarray, img2_np: np.ndarray, weight: float = 0.5) -> Image:
    
    img_new = np.copy(img1_np)

    img_m = np.multiply(img1_np, 1-weight).clip(0, 255).astype(int) + np.multiply(img2_np, weight).clip(0, 255).astype(int)  
    img_new[:, :, 0] = img_m[:, :, 0]
    img_new[:, :, 1] = img_m[:, :, 1]
    img_new[:, :, 2] = img_m[:, :, 2]
            
    return img_new
    
"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
Function to limit the chroma of "img_new" to have an absolute percentage 
difference respect to "U","V" provided by "img_stable" not higher than "alpha"  
"""
def chroma_stabilizer(img_stable: Image, img_new: Image, alpha: float = 0.2, weight: float = 1.0) -> Image:
    
    img1_np = np.asarray(img_stable)
    yuv1 = cv2.cvtColor(img1_np, cv2.COLOR_RGB2YUV)
    y1 = yuv1[:, :, 0]
    u1 = yuv1[:, :, 1]
    v1 = yuv1[:, :, 2]
    
    u1_up = np.multiply(u1, 1 + alpha).clip(0, 255).astype(int)
    v1_up = np.multiply(v1, 1 + alpha).clip(0, 255).astype(int)
    
    u1_dn = np.multiply(u1, 1 - alpha).clip(0, 255).astype(int)
    v1_dn = np.multiply(v1, 1 - alpha).clip(0, 255).astype(int)
        
    img2_np = np.asarray(img_new)
    yuv2 = cv2.cvtColor(img2_np, cv2.COLOR_RGB2YUV)
    
    v2_new = np.copy(yuv2)

    u2 = yuv2[:, :, 1]
    v2 = yuv2[:, :, 2]    
    
    u2_m = array_max_min(u2, u1_up, u1_dn)
    v2_m = array_max_min(v2, v1_up, v1_dn)
        
    v2_new[:, :, 0] = y1
    v2_new[:, :, 1] = u2_m
    v2_new[:, :, 2] = v2_m
    
    if weight < 1.0:
       v2_m = np.multiply(yuv1, 1-weight).clip(0, 255).astype(int) + np.multiply(v2_new, weight).clip(0, 255).astype(int)  
       v2_new[:, :, 0] = v2_m[:, :, 0]
       v2_new[:, :, 1] = v2_m[:, :, 1]
       v2_new[:, :, 2] = v2_m[:, :, 2]
        
    v2_rgb = cv2.cvtColor(v2_new, cv2.COLOR_YUV2RGB)
    
    return Image.fromarray(v2_rgb)

"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
implementation of max() function on numpy array, beacuse this function 
is not available on the base library.  
"""
def array_max(a: np.ndarray, a_max: np.ndarray, dtype: np.dtype = np.int32) -> np.ndarray:
    return np.where(a > a_max, a_max, a).astype(dtype)

"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
implementation of min() function on numpy array, beacuse this function 
is not available on the base library.  
"""
def array_min(a: np.ndarray, a_min: np.ndarray, dtype: np.dtype = np.int32) -> np.ndarray:
    return np.where(a < a_min, a_min, a).astype(dtype)

"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
implementation of min(max()) function on numpy array.
"""

def array_max_min(a: np.ndarray, a_max: np.ndarray, a_min: np.ndarray, dtype: np.dtype = np.int32) -> np.ndarray:
    a_m = array_max(a, a_max, dtype)
    return array_min(a_m, a_min, dtype)

def array_min_max(a: np.ndarray, a_min: np.ndarray, a_max: np.ndarray, dtype: np.dtype = np.int32) -> np.ndarray:
    a_m = array_max(a, a_max, dtype)
    return array_min(a_m, a_min, dtype)
    
"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
Implementation of function chroma_stabilizer() with fixed threshold
of 20% using the Pillow library (slower than chroma_stabilizer)   
"""
def chroma_smoother(img_prv: Image, img: Image) -> Image:

    r2, g2, b2 = img.split()

    img1_up = Image.eval(img_prv, (lambda x: min(x*(1 + 0.20),255)) ) 
    img1_dn = Image.eval(img_prv, (lambda x: max(x*(1 - 0.20), 0)) ) 

    r1_up, g1_up, b1_up = img1_up.split()
    r1_dn, g1_dn, b1_dn = img1_dn.split()

    r_m = ImageMath.eval("convert(max(min(a, c), b), 'L')", a=r1_up, b=r1_dn, c=r2)
    g_m = ImageMath.eval("convert(max(min(a, c), b), 'L')", a=g1_up, b=g1_dn, c=r2)
    b_m = ImageMath.eval("convert(max(min(a, c), b), 'L')", a=b1_up, b=b1_dn, c=r2)

    img_m = Image.merge('RGB', (r_m, g_m, b_m))
    
    img_final = chroma_post_process(img_m, img)
    
    return img_final

"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
Function to copy the chroma parametrs "U", "V", of "img_m" in "orig" 
"""
def chroma_post_process(img_m: Image, orig: Image) -> Image:
    img_np = np.asarray(img_m)
    orig_np = np.asarray(orig)
    img_yuv = cv2.cvtColor(img_np, cv2.COLOR_RGB2YUV)
    # copy the chroma parametrs "U", "V", of "img_m" in "orig" 
    orig_yuv = cv2.cvtColor(orig_np, cv2.COLOR_RGB2YUV)
    orig_copy = np.copy(orig_yuv)
    orig_copy[:, :, 1:3] = img_yuv[:, :, 1:3]
    img_np_new = cv2.cvtColor(orig_copy, cv2.COLOR_YUV2RGB)
    return Image.fromarray(img_np_new)

def chroma_np_post_process(img_np: np.ndarray, orig_np: np.ndarray) -> Image:
    img_yuv = cv2.cvtColor(img_np, cv2.COLOR_RGB2YUV)
    # copy the chroma parametrs "U", "V", of "img_m" in "orig" 
    orig_yuv = cv2.cvtColor(orig_np, cv2.COLOR_RGB2YUV)
    orig_copy = np.copy(orig_yuv)
    orig_copy[:, :, 1:3] = img_yuv[:, :, 1:3]
    return cv2.cvtColor(orig_copy, cv2.COLOR_YUV2RGB)

"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
This function force the average luma of an image to don't be below the value
defined by the parameter "luma_min". The function allow to modify the gamma
of image if the average luma is below the parameter "gamma_luma_min"  
"""
def luma_adjusted_levels(img: Image, luma_min: float = 0, gamma: float = 1.0, gamma_luma_min: float = 0, gamma_alpha: float = 0, gamma_min: float = 0.2, i_min: int = 0, i_max: int = 255) -> Image:
        
    img_np = np.asarray(img)
    
    yuv = cv2.cvtColor(img_np, cv2.COLOR_RGB2YUV)

    y = yuv[:, :, 0]
    u = yuv[:, :, 1]
    v = yuv[:, :, 2]
    
    luma = np.mean(yuv[:,:,0])/255
    
    if luma < luma_min:
        i_alpha = int(255*(luma_min - luma))
    else:
        i_alpha = 0

    yuv_new = np.copy(yuv).clip(i_min, i_max)

    if i_alpha > 1:    
        y_new = np.add(y, i_alpha).clip(i_min, i_max).astype(int)
    else:
        y_new = y
    
    if gamma != 1 and luma < gamma_luma_min:
        if gamma_alpha != 0:
            g_new = max(gamma * pow(luma/gamma_luma_min, gamma_alpha), gamma_min)
        else:
            g_new = gamma
        y_new = np.power(y_new / 255, 1/g_new)
        y_new = np.multiply(y_new, 255).clip(i_min, i_max).astype(int) 

    yuv_new[:, :, 0] = y_new
    yuv_new[:, :, 1] = u
    yuv_new[:, :, 2] = v

    rgb_np = cv2.cvtColor(yuv_new, cv2.COLOR_YUV2RGB)
    return Image.fromarray(rgb_np)

"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
adjust the contrast of an image, color-space: YUV 
"""
def image_contrast(img: Image, cont: float = 1.0, perc: float = 5):
    
    if (cont == 1):
        return img
    
    img_np = np.asarray(img)
    yuv = cv2.cvtColor(img_np, cv2.COLOR_RGB2YUV)
    
    y = yuv[:, :, 0]
    
    y_min = np.percentile(y, perc)
    y_max = np.percentile(y, 100-perc)
    y_fix = np.clip(y, y_min, y_max)
    y_cont = ((y_fix - y_min) * cont / (y_max - y_min))
    
    y_cont = array_min_max(y_cont, 0, 1, np.float64)*255
    
    yuv_new = np.copy(yuv)
    
    yuv_new[:, :, 0] = y_contclip(0,255).astype(int)
    
    img_rgb = cv2.cvtColor(yuv_new, cv2.COLOR_YUV2RGB)
    
    return Image.fromarray(img_rgb)

"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
adjust the brightness of an image, color-space: YUV 
"""
def image_brightness(img: Image, bright: float = 0.0):
    
    if (bright == 0):
        return img
    
    img_np = np.asarray(img)
    yuv = cv2.cvtColor(img_np, cv2.COLOR_RGB2YUV)
    
    y = yuv[:, :, 0]
    
    y_cont = y/255 + bright
    
    y_cont = array_min_max(y_cont, 0, 1, np.float64)*255
    
    yuv_new = np.copy(yuv)
    
    yuv_new[:, :, 0] = y_cont.clip(0,255).astype(int)
    
    img_rgb = cv2.cvtColor(yuv_new, cv2.COLOR_YUV2RGB)
    
    return Image.fromarray(img_rgb)

"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
Simple function adjust hue and decrease saturation/brightness of an image.    
"""
def image_tweak(img: Image, sat: float = 1, bright: float = 0, hue: float = 0) -> Image:

    if (sat == 1 and bright == 0 and hue == 0):
        return img  # non changes
    
    img_np = np.asarray(img)
        
    img_rgb = np_image_tweak(img_np, sat, bright, hue)
    
    return Image.fromarray(img_rgb,'RGB').convert('RGB') 

def np_image_tweak(img_np: np.ndarray, sat: float = 1, bright: float = 0, hue: float = 0) -> Image:

    if (sat == 1 and bright == 0 and hue == 0):
        return img_np  # non changes
       
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    
    hsv[:, :, 0] = hsv[:, :, 0] * min(max(1 + hue, 0),2)
    hsv[:, :, 1] = hsv[:, :, 1] * min(max(sat, 0), 1)
    hsv[:, :, 2] = hsv[:, :, 2] * min(max(1 + bright, 0), 1)
               
    img_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    return img_rgb

"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
This function is an extension of the Tweak() function available in Hybrid with
the possibility to change also the gamma of a video clip.    
"""
def tweak(clip: vs.VideoNode, hue: float = 0, sat: float = 1, bright: float = 0, cont: float = 1, gamma: float = 1, coring: bool = True) -> vs.VideoNode:

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

"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
get the value of average brightness of an image
"""
def get_pil_brightness(img: Image) -> float:
    img_np = np.asarray(img)
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    brightness = np.mean(hsv[:,:, 2])
    return (brightness/255)

"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
get the value of average luma of an image
"""
def get_pil_luma(img: Image) -> float:
    img_np = np.asarray(img)
    yuv = cv2.cvtColor(img_np, cv2.COLOR_RGB2YUV)
    luma = np.mean(yuv[: ,:, 0])
    return (luma/255)

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
def vs_chroma_stabilizer_ex(clip: vs.VideoNode = None, nframes: int = 5, mode: str = "A", sat: float = 1.0, tht: int = 0, weight: float = 0.5, tht_scen: float = 0.8, algo: int = 0) -> vs.VideoNode:

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
        clip_rgb = _average_clips_ex(clip=clip, weight_list=weight_list, sat=sat, tht=tht, weight=weight, tht_scen=tht_scen)
    else:
        clip_rgb = _average_frames_ex(clip=clip, weight_list=weight_list, sat=sat, tht=tht, weight=weight, tht_scen=tht_scen)
         
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
Temporal luma limiter: the function will limit the luma of "cur_img" to have an 
absolute percentage deviation respect to "prv_img" not higher than "alpha"  
"""
def _chroma_temporal_limiter(cur_img: Image, prv_img: Image, alpha: float = 0.05) -> Image:
    
    img1_np = np.asarray(prv_img)
    yuv1 = cv2.cvtColor(img1_np, cv2.COLOR_RGB2YUV)
    u1 = yuv1[:, :, 1]
    v1 = yuv1[:, :, 2]
    
    u1_up = np.multiply(u1, 1 + alpha)
    u1_dn = np.multiply(u1, 1 - alpha)
    
    v1_up = np.multiply(v1, 1 + alpha)
    v1_dn = np.multiply(v1, 1 - alpha)
            
    img2_np = np.asarray(cur_img)
    yuv2 = cv2.cvtColor(img2_np, cv2.COLOR_RGB2YUV)
    
    yuv_new = np.copy(yuv2)

    u2 = yuv2[:, :, 1]
    v2 = yuv2[:, :, 2]
    
    u2_m = array_max_min(u2, u1_up, u1_dn)
    v2_m = array_max_min(v2, v1_up, v1_dn)
        
    yuv_new[:, :, 1] = u2_m
    yuv_new[:, :, 2] = v2_m
    
    rgb_new = cv2.cvtColor(yuv_new, cv2.COLOR_YUV2RGB)
    
    return Image.fromarray(rgb_new)

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
def _average_frames_ex(clip: vs.VideoNode = None, weight_list: list = None, sat: float = 1.0, tht: int = 0, weight: float = 0.2, tht_scen: float = 0.8) -> vs.VideoNode:
    
    def smooth_frame(n, f, clip_base: vs.VideoNode = None, weight_list: list = None, sat: float = 1.0, tht: int = 0, weight: float = 0.2, tht_scen: float = 0.8) -> vs.VideoFrame:
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
            img_f.append(restore_color(img_base, frame_to_image(clip_base.get_frame(Ni)), sat, tht, weight, tht_scen))
        img_f.append(img_base)
        for i in range(0, Nh):
            Ni = n + (i+1)
            img_f.append(restore_color(img_base, frame_to_image(clip_base.get_frame(Ni)), sat, tht, weight, tht_scen))
        img_m = _color_temporal_stabilizer(img_f, weight_list)
        return image_to_frame(img_m, f_out)    
    clip = clip.std.ModifyFrame(clips=[clip], selector=partial(smooth_frame, clip_base=clip, weight_list=weight_list, sat=sat, tht=tht, weight=weight, tht_scen=tht_scen))
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
def _average_clips_ex(clip: vs.VideoNode = None, weight_list: list = None, sat: float = 1.0, tht: int = 0, weight: float = 0.2, tht_scen: float = 0.8) -> vs.VideoNode:

    max_frames = len(weight_list)
    clips = list()
    clip_yuv = clip.resize.Bicubic(format=vs.YUV420P8, matrix_s="709", range_s="limited", dither_type="error_diffusion")
    Nh = round((max_frames-1)/2)
    for i in range(0, Nh):
        Ni = -(Nh - i)
        clip_i = vs_get_clip_frame(clip=clip, nframe=Ni)
        clip_i = vs_recover_clip_color(clip=clip_i, clip_color=clip, sat=sat, tht=tht, weight=weight, tht_scen=tht_scen, return_mask=False)
        clips.append(clip_i.resize.Bicubic(format=vs.YUV420P8, matrix_s="709", range_s="limited", dither_type="error_diffusion"))
    clips.append(clip_yuv)
    for i in range(0, Nh):
        Ni = i + 1
        clip_i = vs_get_clip_frame(clip=clip, nframe=Ni)
        clip_i = vs_recover_clip_color(clip=clip_i, clip_color=clip, sat=sat, tht=tht, weight=weight, tht_scen=tht_scen, return_mask=False)
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
    weight: float = 0.2, tht_scen: float = 0.8, return_mask: bool = False) -> vs.VideoNode:
  
    def color_frame(n, f, sat: float = 1.0, tht: int = 0, weight: float = 0.2, tht_scen: float = 0.8, return_mask: bool = False):        
        f_out = f[0].copy()        
        if n < 15:
            return f_out        
        img_gray = frame_to_image(f[0])
        img_color = frame_to_image(f[1])
        img_restored = restore_color(img_color, img_gray, sat, tht, weight, tht_scen, return_mask)
        return image_to_frame(img_restored, f_out)    
    clip = clip.std.ModifyFrame(clips=[clip, clip_color], selector=partial(color_frame, sat=sat, tht=tht, weight=weight, tht_scen=tht_scen, return_mask=return_mask))
    return clip
    
"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
Temporal color stabilizer the UV chroma of current frame are averaged with the
values of previous "nframes"  
"""
def _color_temporal_stabilizer(img_f: list, weight_list: list = None) -> Image:
    
    nframes = len(weight_list)
    
    Nh = round((nframes-1)/2)
    
    img_new = np.copy(np.asarray(img_f[Nh]))
    
    yuv_new = cv2.cvtColor(img_new, cv2.COLOR_RGB2YUV)
    
    weight: float = weight_list[Nh]/100.0

    yuv_m = np.multiply(yuv_new, weight)
    
    for i in range (0, Nh):
        yuv_i = cv2.cvtColor(np.asarray(img_f[i]), cv2.COLOR_RGB2YUV) 
        weight: float = weight_list[i]/100.0
        yuv_m += np.multiply(yuv_i, weight)
    for i in range (Nh+1, nframes):
        yuv_i = cv2.cvtColor(np.asarray(img_f[i]), cv2.COLOR_RGB2YUV) 
        weight: float = weight_list[i]/100.0
        yuv_m += np.multiply(yuv_i, weight)
        
    yuv_new[:, :, 1] = yuv_m[:, :, 1]
    yuv_new[:, :, 2] = yuv_m[:, :, 2]
            
    return Image.fromarray(cv2.cvtColor(yuv_new, cv2.COLOR_YUV2RGB))
 
def restore_color(img_color: Image = None, img_gray: Image = None, sat: float=1.0, tht: int = 80, weight: float = 0, tht_scen=0.8, return_mask: bool=False) -> Image:
      
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
        return np_gray
    
    mask_rgb = np_gray.copy();
    
    for i in range(3):
        mask_rgb[:,:,i] = hsv_mask
    
    if return_mask:
        return Image.fromarray(mask_rgb,'RGB').convert('RGB') 
        
    np_restored = np_image_mask_merge(np_gray, np_color_sat, mask_rgb)
    
    if weight > 0:
        np_restored = np_weighted_merge(np_restored, np_gray, weight)
    
    return Image.fromarray(np_restored,'RGB').convert('RGB') 

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

