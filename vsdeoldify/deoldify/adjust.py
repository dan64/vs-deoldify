import vapoursynth as vs
import math

def combine_models(clipa: vs.VideoNode = None, clipb: vs.VideoNode = None, sat: list = [1,1], hue: list = [0,0], method: int = 0, clipb_weight: float = 0.0) -> vs.VideoNode:

    if clipa is not None:
        clipa = tweak(clipa, hue=hue[0], sat=sat[0])
        if clipb is None: return clipa
    
    if clipb is not None:
        clipb = tweak(clipb, hue=hue[1], sat=sat[1])
        if clipa is None: return clipb
        
    if method == 0:
        return vs.core.std.Merge(clipa, clipb, weight=clipb_weight)
    else:
        raise vs.Error("deoldify: only dd_method=0 is supported")
    

def tweak(clip: vs.VideoNode, hue: int = 0, sat: int = 1, bright=None, cont=None, coring=True):

    if sat == 1 and hue == 0:
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

    if bright is not None or cont is not None:
        bright = 0.0 if bright is None else bright
        cont = 1.0 if cont is None else cont

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
            
    # convert the clip format for deoldify
    return clip.resize.Bicubic(format=vs.RGB24, matrix_in_s="709", range_s="limited", dither_type="error_diffusion") 

    