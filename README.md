# Hybrid Automatic Video Colorizer (aka HAVC)
A Deep Learning based [ VapourSynth](https://www.vapoursynth.com/) filter for colorizing and restoring old images and video, based on the following projects: [DeOldify](https://github.com/jantic/DeOldify)
,  [DDColor](https://github.com/HolyWu/vs-ddcolor), [Colorization](https://github.com/richzhang/colorization), [Deep Exemplar based Video Colorization](https://github.com/zhangmozhe/Deep-Exemplar-based-Video-Colorization), [DeepRemaster](https://github.com/satoshiiizuka/siggraphasia2019_remastering) and [ColorMNet](https://github.com/yyang181/colormnet). The project  [Colorization](https://github.com/richzhang/colorization) includes 2 models: _Real-Time User-Guided Image Colorization with Learned Deep Priors_ (Zhang, 2017) and _Colorful Image Colorization_ (Zhang, 2016). These 2 models has been added as alternative models (named: _siggraph17_, _eccv16_) to DDColor .

The Vapoursynth filter version has the advantage of coloring the images directly in memory, without the need to use the filesystem to store the video frames. 

For this filter is available a [User Guide](https://github.com/dan64/vs-deoldify/blob/main/documentation/HAVC%20User%20Guide.pdf) which provides useful tips and detailed explanations regarding the filter functions and usage. It is strongly recommended reading it before using the filter.

The filter (_HAVC_ in short) is able to combine the results provided by _DeOldify_ and _DDColor_ (_Colorization_), which are some of the best models available for coloring pictures, providing often a final colorized image that is better than the image obtained from the individual models.  But the main strength of this filter is the addition of specialized filters to improve the quality of videos obtained by using these color models and the possibility to improve further the stability by using these models as input to [Deep Exemplar based Video Colorization](https://github.com/zhangmozhe/Deep-Exemplar-based-Video-Colorization) model (_DeepEx_ in short), [DeepRemaster](https://github.com/satoshiiizuka/siggraphasia2019_remastering) and  [ColorMNet](https://github.com/yyang181/colormnet).

  _DeepEx_, _DeepRemaster_ and _ColorMNet_ are exemplar-based video colorization models, which allow to colorize a movie starting from an external-colored reference image. They allow to colorize a Video in sequence based on the colorization history, enforcing its coherency by using a temporal consistency loss. _ColorMNet_ is more recent and advanced respect to _DeepEx_ and it is suggested to use it as default exemplar-based model.

  _DeepRemaster_ has the interesting feature to be able store the reference images, so that is able to manages situations where the reference images not are synchronized with the movie to colorize. Conversely _ColorMNet_ is not storing the full reference frame image (like _DeepRemaster_) but it stores only the key points (e.g., representative pixels in each frame). This imply that the colored frames could have some colors that are very different from the reference image. _DeepRemaster_ has not this problem since it stores the full reference image. Unfortunately, the number of reference images that DeepRemaster is able to use depends on GPU memory and power, because the time required for inference increase with the number of reference images provided. Instead _ColorMNet_ has some interpolation capability while _DeepRemaster_ is very basic and is unable to properly colorize a frame if is missing a reference image very similar and it need a lot of reference images to be able to properly colorize a movie (the time resolution of _DeepRemaster_ is 15 frames). So, the choice of which exemplar-based video colorization model to use depends on the source to colorize and the number of reference image available. 

## Quick Start

This filter is distributed with the torch package provided with the **Hybrid Windows Addons**. To use it on Desktop (Windows) it is necessary install [Hybrid](https://www.selur.de/downloads) and the related [Addons](https://drive.google.com/drive/folders/1vC_pxwxL0o8fjmg8Okn0RA5rsodTcv9G?usp=drive_link). **Hybrid** is a Qt-based frontend for other tools (including this filter) which can convert most input formats to common audio & video formats and containers. It represent the  easiest way to colorize images with the HAVC filter using [VapourSynth](https://www.vapoursynth.com/).  In the folder _documentation_ is available a [User Guide](https://github.com/dan64/vs-deoldify/blob/main/documentation/HAVC%20User%20Guide.pdf) that provides detailed information on how to install Hybrid and use it to colorize videos. The Guide also provides tips on how to improve the final quality of colored movies.     


## Dependencies
- [PyTorch](https://pytorch.org/get-started) 2.1.1 or later
- [VapourSynth](http://www.vapoursynth.com/) R62 or later
- [MiscFilters.dll](https://github.com/vapoursynth/vs-miscfilters-obsolete) Vapoursynth's Miscellaneous Filters


## Installation
```
pip install vsdeoldify-x.x.x-py3-none-any.whl
```
with the version [4.0.0](https://github.com/dan64/vs-deoldify/releases/tag/v4.0.0) of HAVC has been released a modified version of DDColor to manage the Scene Detection properties available in the input clip, this version can be installed with the command:

```
pip install vsddcolor-1.0.1-py3-none-any.whl.zip
```

with the version 4.5.0 of HAVC has been introduced the support to ColorMNet. All the necessary packages to use ColorMNet are included in Hybrid's torch add-on package.  For a manual installation not using Hybrid, it is necessary to install all the packages reported in the project page of [ColorMNet](https://github.com/yyang181/colormnet). To simplify the installation,  in the release [4.5.0](https://github.com/dan64/vs-deoldify/releases/tag/v4.5.0) of this filter is available as asset the **spatial_correlation_sampler** package compiled against CUDA 12.4, python 3.12 and torch. To install it is necessary to unzip the following archive (using the nearest torch version available in the host system):

```
spatial_correlation_sampler-0.5.0-py312-cp312-win_amd64_torch-x.x.x.whl.zip 
```

in the Library packages folder: .\Lib\site-packages\

## Models Download
The models are not installed with the package, they must be downloaded from the Deoldify website at: [completed-generator-weights](https://github.com/jantic/DeOldify#completed-generator-weights).

The models to download are:

- ColorizeVideo_gen.pth
- ColorizeStable_gen.pth
- ColorizeArtistic_gen.pth

The _model files_ have to be copied in the **models** directory usually located in:

.\Lib\site-packages\vsdeoldify\models

To use ColorMNet it also necessary to download the file [DINOv2FeatureV6_LocalAtten_s2_154000.pth](https://github.com/yyang181/colormnet/releases/download/v0.1/DINOv2FeatureV6_LocalAtten_s2_154000.pth) and save it in

.\Lib\site-packages\vsdeoldify\colormnet\weights

With the version 5.0 of HAVC has been added the model [DeepRemaster](https://github.com/satoshiiizuka/siggraphasia2019_remastering), for using it is necessary to download the file [remasternet.pth.tar ](http://iizuka.cs.tsukuba.ac.jp/data/remasternet.pth.tar) (is not a tar, just a "pth" renamed as "pth.tar") and copy it in: ".\Lib\site-packages\vsdeoldify\remaster\model".

At the first usage it is possible that are automatically downloaded by torch the neural networks: **resnet101** and **resnet34**, and starting with the release 4.5.0:  **resnet50**, **resnet18**, **dinov2_vits14_pretrain** and the folder **facebookresearch_dinov2_main** 

So don't be worried if at the first usage the filter will be very slow to start, at the initialization are loaded almost all the _Fastai_ and _PyTorch_ modules and the resnet networks.

It is possible specify the destination directory of networks used by torch, by using the function parameter **torch\_hub\_dir**, if this parameter is set to **None**, the files will be downloaded in the _torch's cache_ dir, more details are available at: [caching-logic ](https://pytorch.org/docs/stable/hub.html#caching-logic).

The models used by **DDColor** can be installed with the command

```
python -m vsddcolor
```

The models for **Deep-Exemplar based Video Colorization.** can be installed by downloading  the file **colorization_checkpoint.zip** available in: [inference code](https://github.com/zhangmozhe/Deep-Exemplar-based-Video-Colorization/releases/tag/v1.0). 
 
The archive  **colorization_checkpoint.zip** have to be unziped in: .\Lib\site-packages\vsdeoldify\deepex

## Usage
```python
# loading plugins
core.std.LoadPlugin(path="MiscFilters.dll")
import vsdeoldify as havc

# changing range from limited to full range for HAVC
clip = core.resize.Bicubic(clip, range_in_s="limited", range_s="full")
# setting color range to PC (full) range.
clip = core.std.SetFrameProps(clip=clip, _ColorRange=0)
# adjusting color space from YUV420P16 to RGB24
clip = core.resize.Bicubic(clip=clip, format=vs.RGB24, matrix_in_s="709", range_s="full")


# DeOldify with DDColor, Preset = "fast"
clip = havc.HAVC_main(clip=clip, Preset="fast")
# DeOldify only model
clip = havc.HAVC_colorizer(clip, method=0)
# DDColor only model
clip = havc.HAVC_colorizer(clip, method=1)

# To apply video color stabilization filters to colored clip 
clip = havc.HAVC_stabilizer(clip, dark=True, smooth=True, stab=True)

# Simplest way to use Presets
clip = havc.HAVC_main(clip=clip, Preset="fast", ColorFix="violet/red", ColorTune="medium", ColorMap="none")

# ColorMNet model using HAVC as input for the reference frames
clip = havc.HAVC_main(clip=clip, EnableDeepEx=True, ScThreshold=0.1)

# changing range from full to limited range for HAVC
clip = core.resize.Bicubic(clip, range_in_s="full", range_s="limited")
```

See `__init__.py` for the description of the parameters.

**NOTES**: 

- In the _DDColor_ version included with **HAVC** the parameter _input_size_ has changed name in _render_factor_ because were changed the range of values of this parameter to be equivalent to _render_factor_ in _DeOldify_, the relationship between these 2 parameters is the following:

```
input_size = render_factor * 16
``` 

- In the modified version of _DDColor_  1.0.1 was added the boolean parameter _scenechange_, if this parameter is set to _True_, will be colored only the frames tagged as scene change.  

- In the folder [samples](https://github.com/dan64/vs-deoldify/tree/main/samples) there are some clips and reference images that can be used to test the filter. The clips _sample_colored_sync.mp4_ and _sample_colored_async.mp4_ are useful to test the new video restore functionality added in HAVC 5.0 (described in the User Guide). The clip _sample_colored_sync.mp4_ is fully in sync with the clip _sample_bw_.mp4 and any of the exemplar-based models can be used to colorize it, while the clip _sample_colored_async.mp4_ is not in sync and only _DeepRemaster_ is able to properly colorize the movie.   

## Filter Usage

The filter was developed having in mind to use it mainly to colorize movies. Both DeOldify and DDcolor are good models for coloring pictures (see the _Comparison of Models_). But when are used for coloring movies they are introducing artifacts that usually are not  noticeable in the images.  Especially in dark scenes both DeOldify and DDcolor are not able to understand what it is the dark area and what color to give it, they often decide to color these dark areas with blue, then in the next frame this area could become red and then in the next frame return to blue, introducing a flashing psychedelic effect when all the frames are put in a movie.
To try to solve this problem has been developed _pre-_ and _post-_ process filters.  It is possible to see them in the Hybrid screenshot below.

![Hybrid Coloring page](https://github.com/dan64/vs-deoldify/blob/main/hybrid_setup/Model_D%2BD_filters.JPG)  

The main filters introduced are:

**Chroma Smoothing** : This filter allows to reduce the _vibrancy_ of colors assigned by DeOldify/DDcolor by using the parameters _de-saturation_ and _de-vibrancy_ (the effect on _vibrancy_ will be visible only if the option **chroma resize** is enabled, otherwise this parameter has effect on the _luminosity_). The area impacted by the filter is defined by the thresholds dark/white. All the pixels with luma below the dark threshold will be impacted by the filter, while the pixels above the white threshold will be left untouched. All the pixels in the middle will be gradually impacted depending on the luma value.

**Chroma Stabilization**: This filter will try to stabilize the frames' colors. As explained previously since the frames are colored individually, the colors can change significantly from one frame to the next, introducing a disturbing psychedelic flashing effect. This filter try to reduce this by averaging the chroma component of the frames. The average is performed using a number of frames specified in the _Frames_ parameter. 
Are implemented 2 averaging methods: 

1. _Arithmetic average_: the current frame is averaged using equal weights on the past and future frames
2. _Weighted average_: the current frame is averaged using a weighed mean of the past and future frames, where the weight decrease with the time (far frames have lower weight respect to the nearest frames). 

As explained previously the stabilization is performed by averaging the past/future frames. Since the non matched areas of past/future frames are _gray_ because is missing in the past/future the _color information_, the filter will apply a _color restore_ procedure that fills the gray areas with the pixels of current frames (eventually de-saturated with the parameter "sat"). The image restored in this way is blended with the non restored image using the parameter "weight". The gray areas are selected by the threshold parameter "tht". All the pixels in the HSV color space with "S" < "tht" will be considered gray. If is detected a scene change (controlled by the parameter "tht_scen"), the _color restore_ is not applied.  

**DDColor Tweaks**: This filter is available only for DDColor and has been added because has been observed that the DDcolor's _inference_ is quite poor on dark/bright scenes depending on the luma value. This filter will force the luma of input image to don't be below the threshold defined by the parameter _luma_min_.  Moreover this filter allows to apply a dynamic gamma correction. The gamma adjustment will be applied when the average luma is below the parameter _gamma_luma_min_. A _gamma_ value > 2.0 improves the DDColor stability on bright scenes, while a _gamma_ < 1 improves the DDColor stability on  dark scenes. 

### Chroma Adjustment

Unfortunately when are applied to movies the color models are subject to assign unstable colors to the frames especially on the red/violet chroma range. This problem is more visible on DDColor than on DeOldify.
To mitigate this issue was necessary to implement some kind of chroma adjustment. This adjustment allows to de-saturate all the colors included in a given color range. The color range must be specified in the HSV color space. This color space is useful because all the chroma is represented by only the parameter "Hue". In this color space the colors are specified in degree (from 0 to 360), as shown in the [DDeoldify Hue Wheel](https://github.com/dan64/vs-deoldify/blob/main/hybrid_setup/ddeoldify_hue_wheel.jpg).
It is possible to apply this adjustment on all filters described previously.
Depending on the filter the adjustment can be enabled using the following syntax:

```
chroma_range = "hue_start:hue_end" or "hue_wheel_name"
``` 
for example this assignment: 

```
chroma_range = "290:330,rose"
``` 
specify the range of hue colors: 290-360, because "rose" is [hue wheel name](https://github.com/dan64/vs-deoldify/blob/main/hybrid_setup/ddeoldify_hue_wheel.jpg) that correspond to the range:330-360.

It is possible to specify more ranges by using the comma "," separator.

When the de-saturation information is not already available in the filter's parameters, it necessary to use the following syntax:

```
chroma_adjustment = "chroma_range|sat,weight"
``` 

in this case it is necessary to specify also the de-saturation parameter "sat" and the blending parameter "weight".

for example with this assignment: 

```
chroma_range = "300:340|0.4,0.2"
``` 

the hue colors in the range 300-340 will be de-saturated by the amount 0.4 and the final frame will be blended by applying a 20% de-saturation of 0.4 an all the pixels (if weight=0, no blending is applied).  

To simplify the usage of this filter has been added the Preset _ColorFix_ which allows to fix a given range of chroma combination. The strength of the filter is controlled by the the Preset _ColorTune_.

#### Color Mapping

Using an approach similar to _Chroma Adjustment_ has been introduced the possibility to remap a given gange of colors in another chroma range. This remapping is controlled by the Preset _ColorMap_. For example the preset "blue->brown" allows to remap all the chroma combinations of _blue_ in the color _brown_. It is not expected that this filter can be applied on a full movie, but it could be useful to remap the color on some portion of a movie.

In the [HAVC User Guide](https://github.com/dan64/vs-deoldify/blob/main/documentation/HAVC%20User%20Guide.pdf)  are provided useful tips on how to use both the _Chroma Adjustment_ and _Color Mapping_ features provided by this filter. 
  

### Merging the models

As explained previously, this filter is able to combine the results provided by DeOldify and DDColor, to perform this combination has been implemented 6 methods:

0. _DeOldify_ only coloring model.

1. _DDColor_ only color model.

2. _Simple Merge_: the frames are combined using a _weighted merge_, where the parameter _merge_weight_ represent the weight assigned to the frames provided by the DDcolor model. 

3. _Constrained Chroma Merge_:  given that the colors provided by DeOldify's _Video_ model are more conservative and stable than the colors obtained with DDcolor. The frames are combined by assigning a limit to the amount of difference in chroma values between DeOldify and DDcolor. This limit is defined by the parameter _threshold_. The limit is applied to the frame converted to "YUV". For example when threshold=0.1, the chroma    values "U","V" of DDcolor frame will be constrained to have an absolute percentage difference respect to "U","V" provided by DeOldify not higher than 10%.  If _merge_weight_ is < 1.0, the chroma limited DDColor frames will be will be merged again with the frames of DeOldify using the _Simple Merge_.

 4. _Luma Masked Merge_: the behaviour is similar to the method _Adaptive Luma Merge_. With this method the frames are combined using a _masked merge_. The pixels of DDColor's frame with luma < _luma_limit_  will be filled with the (de-saturated) pixels of DeOldify, while the pixels above the _white_limit_ threshold will be left untouched. All the pixels in the middle will be gradually replaced depending on the luma value. If the parameter  _merge_weight_ is < 1.0, the resulting masked frames will be merged again with the non de-saturated frames of DeOldify using the _Simple Merge_.

5. _Adaptive Luma Merge_: given that the DDcolor performance is quite bad on dark scenes, with this method the images are combined by decreasing the weight assigned to DDcolor frames when the luma is below the _luma_threshold_. For example with: luma_threshold = 0.6 and alpha = 1, the weight assigned to DDcolor frames will start to decrease linearly when the luma < 60% till _min_weight_. For _alpha_=2, the weight begins to decrease quadratically.      

The merging methods 2-5 are leveraging on the fact that usually the DeOldify _Video_ model provides frames which are more stable, this feature is exploited to stabilize also DDColor. The methods 3 and 4 are similar to _Simple Merge_, but before the merge with _DeOldify_ the _DDColor_ frame is limited in the chroma changes (method 3) or limited based on the luma (method 4). The method 5 is a _Simple Merge_ where the weight decrease with luma. 

## Comparison of Models

Taking inspiration from the article published on Habr: [Mode on: Comparing the two best colorization AI's](https://habr.com/en/companies/ruvds/articles/568426/). It was decided to use it to get the refence images and the images obtained using the [ColTran](https://github.com/google-research/google-research/tree/master/coltran) model, to extend the analysis with the models implemented in the **HAVC** filter.

The added models are:

**D+D**: DeOldify (with model _Video_ & render_factor = 24) + DDColor (with model _Artistic_ and render_factor = 24)
![Hybrid D+D](https://github.com/dan64/vs-deoldify/blob/main/hybrid_setup/Model_D%2BD.JPG)  

**DD**:  DDColor (with model _Artistic_ and and render_factor = 24 equivalent to input_size = 384)
![Hybrid_DD](https://github.com/dan64/vs-deoldify/blob/main/hybrid_setup/Model_DD.JPG)

**DS**: DeOldify (with model _Stable_ & render_factor =24)
![Hybrid D+D](https://github.com/dan64/vs-deoldify/blob/main/hybrid_setup/Model_DS.JPG)  

**DV**: DeOldify (with model _Video_ & render_factor = 24)
![Hybrid D+D](https://github.com/dan64/vs-deoldify/blob/main/hybrid_setup/Model_DV.JPG)  

**T241**:  ColTran + TensorFlow 2.4.1 model as shown in [Habr](https://habr.com/en/companies/ruvds/articles/568426/)

### Comparison Methodology

To compare the models I decided to use a metric being able to consider the _perceptual non-uniformities_ in the evaluation of color difference between images. These non-uniformities are important because the human eye is more sensitive to certain colors than others.  Over time, The International Commission on Illumination (**CIE**) has proposed increasingly advanced measurement models to measure the color distance taking into account the _human color perception_, that they called **dE**. One of the most advanced is the [CIEDE2000](https://en.wikipedia.org/wiki/Color_difference#CIEDE2000) method, that I decided to use as _color similarity metric_ to compare the models. The final results are shown in the table below (test image can be seen by clicking on the test number)

### Test Set #1

| Test # | D+D | DD | DS | DV  | T241 |
|---|---|---|---|---|---|
|[01](https://github.com/dan64/vs-deoldify/blob/main/test_images/Image_01_test.jpg) | 10.7 | **8.7** | 8.8 | 12.7 | 15.7 |
|[02](https://github.com/dan64/vs-deoldify/blob/main/test_images/Image_02_test.jpg) | 11.8 | **11.7** | 12.7 | 12.7 | 15.9 |
|[03](https://github.com/dan64/vs-deoldify/blob/main/test_images/Image_03_test.jpg) | 5.5 | **3.8** | 5.6 | 7.6 | 9.9 |
|[04](https://github.com/dan64/vs-deoldify/blob/main/test_images/Image_04_test.jpg) | 6.2 | 8.5 | **4.6** | 5.3 | 9.0 |
|[05](https://github.com/dan64/vs-deoldify/blob/main/test_images/Image_05_test.jpg) | **6.6** | 8.4 | 8.8 | 8.6 | 12.5 |
|[06](https://github.com/dan64/vs-deoldify/blob/main/test_images/Image_06_test.jpg) | 10.2 | **9.9** | 10.6 | 11.2 | 16.4 |
|[07](https://github.com/dan64/vs-deoldify/blob/main/test_images/Image_07_test.jpg) |**6.5** | 6.7 | 6.8 | 7.7 | 10.2 |
|[08](https://github.com/dan64/vs-deoldify/blob/main/test_images/Image_08_test.jpg) | 6.7 | **6.4** | 7.5 | 8.3 | 9.9 |
|[09](https://github.com/dan64/vs-deoldify/blob/main/test_images/Image_09_test.jpg) | 11.7 | **11.7** | 15.2 | 13.8 | 16.5 |
|[10](https://github.com/dan64/vs-deoldify/blob/main/test_images/Image_10_test.jpg) | **7.8** | 8.0 | 9.1 | 8.4 | 9.5 |
|[11](https://github.com/dan64/vs-deoldify/blob/main/test_images/Image_11_test.jpg) | **7.5** | 8.0 | 8.0 | 7.8 | 14.8 |
|[12](https://github.com/dan64/vs-deoldify/blob/main/test_images/Image_12_test.jpg) | 7.7 | **7.6** | 8.6 | 7.8 | 13.7 |
|[13](https://github.com/dan64/vs-deoldify/blob/main/test_images/Image_13_test.jpg) | **11.8** | 11.9 | 14.2 | 13.7 | 16.8 |
|[14](https://github.com/dan64/vs-deoldify/blob/main/test_images/Image_14_test.jpg) | 5.3 | 5.2 | **4.4** | 5.3 | 7.2 |
|[15](https://github.com/dan64/vs-deoldify/blob/main/test_images/Image_15_test.jpg) | 8.2 | **7.3** | 10.7 | 10.6 | 15.7 |
|[16](https://github.com/dan64/vs-deoldify/blob/main/test_images/Image_16_test.jpg) | 12.0 | 12.3 | **9.8** | 12.7 | 19.7 |
|[17](https://github.com/dan64/vs-deoldify/blob/main/test_images/Image_17_test.jpg) | 11.1 | **10.2** | 11.6 | 12.4 | 16.7 |
|[18](https://github.com/dan64/vs-deoldify/blob/main/test_images/Image_18_test.jpg) | **6.7** | 9.3 | 7.2 | 8.6 | 13.1 |
|[19](https://github.com/dan64/vs-deoldify/blob/main/test_images/Image_19_test.jpg) | **3.7** | 4.4 | 4.7 | 3.9 | 4.6 |
|[20](https://github.com/dan64/vs-deoldify/blob/main/test_images/Image_20_test.jpg) | 8.7 | 10.1 | **6.9** | 9.2 | 11.0 |
|[21](https://github.com/dan64/vs-deoldify/blob/main/test_images/Image_21_test.jpg) | **6.9** | 6.9 | 8.1 | 8.4 | 10.4 |
|[22](https://github.com/dan64/vs-deoldify/blob/main/test_images/Image_22_test.jpg) | **11.5** | 11.8 | 13.3 | 12.2 | 12.7 |
|[23](https://github.com/dan64/vs-deoldify/blob/main/test_images/Image_23_test.jpg) | **5.6** | 7.1 | 11.4 | 8.8 | 11. |
|**Avg(dE)** | **8.3** | **8.5** | **9.1** | **9.5** | **12.7** |

   
The calculation of **dE** with the  **CIEDE2000** method was obtained by leveraging on the computational code available in [ColorMine](https://github.com/MasterPieceCode/Mozaic/tree/master/ColorMine).

As it is possible to see the model that performed better is the **D+D** model (which I called _HAVC ddelodify_ because is using both _DeOldify_ and _DDColor_). This model was the best model in 10 tests out of 23. Also the **DD** model performed well but there were situations where the **DD** model provided quite bad colorized images like in [Test #23](https://github.com/dan64/vs-deoldify/blob/main/test_images/Image_23_test.jpg) and the combination with the DeOldify allowed to significantly improve the final image. In effect the average distance of **DD** was **8.5** while for **DV** was **9.5**, given that the 2 models were weighted at 50%, if the images were positively correlated a value **9.0** would have been expected, instead the average distance measured for **D+D** was **8.3**, this implies that the 2 models were able to compensate each other. 
Conversely, the **T241** was the model that performed worse with the greatest average difference in colors. Finally, the quality of DeOldify models was similar, being **DS** slightly better than **DV** (as expected).

###  Tests Set #2

Given the goodness of  **CIEDE2000** method to provide a reliable estimate of _human color perception_, I decided to provide an additional set of tests including some of the cases not considered previously.

The models added are:

**DA**: DeOldify (with model _Artistic_ & render_factor = 30) 

**DDs**: DDColor (with model _ModelScope_ and input_size = 384)

**DS+DD**: DeOldify (with model _Stable_ & render_factor = 30) + DDColor (with model _Artistic_ and render_factor  = 24)

**DA+DDs**: DeOldify (with model _Artistic_ & render_factor = 30) + DDColor (with model _ModelScope_ and render_factor  = 24)

**DA+DD**: DeOldify (with model _Artistic_ & render_factor = 30) + DDColor (with model _Artistic_ and render_factor  = 24)

The results of this additional tests set are shown in the table below (test image can be seen by clicking on the test number)

| Test # | DS+DD | DA+DDs | DA+DD | DDs  | DA |
|---|---|---|---|---|---|
|[01](https://github.com/dan64/vs-deoldify/blob/main/test_images_ex/Image_01_test_ex.jpg) | 7.7 | **7.5** | 8.2 | 8.2 | 8.6 |
|[02](https://github.com/dan64/vs-deoldify/blob/main/test_images_ex/Image_02_test_ex.jpg) | 11.8 | **11.4** | 11.9 | 11.6 | 13.2 |
|[03](https://github.com/dan64/vs-deoldify/blob/main/test_images_ex/Image_03_test_ex.jpg) | 4.5 | 4.2 | **3.9** | 4.5 | 4.2 |
|[04](https://github.com/dan64/vs-deoldify/blob/main/test_images_ex/Image_04_test_ex.jpg) | 5.9 | **5.1** | 6.0 | 6.6 | 5.9 |
|[05](https://github.com/dan64/vs-deoldify/blob/main/test_images_ex/Image_05_test_ex.jpg) | **6.4** | 6.5 | 6.7 | 9.5 | 9.0 |
|[06](https://github.com/dan64/vs-deoldify/blob/main/test_images_ex/Image_06_test_ex.jpg) | 10.0 | 10.0 | 10.3 | **9.5** | 11.4 |
|[07](https://github.com/dan64/vs-deoldify/blob/main/test_images_ex/Image_07_test_ex.jpg) | **6.1** | 7.3 | 6.6 | 8.1 | 8.0 |
|[08](https://github.com/dan64/vs-deoldify/blob/main/test_images_ex/Image_08_test_ex.jpg) | **6.2** | 8.1 | 7.3 | 8.1 | 9.4 |
|[09](https://github.com/dan64/vs-deoldify/blob/main/test_images_ex/Image_09_test_ex.jpg) | 12.7 | **11.3** | 11.5 | 12.5 | 13.3 |
|[10](https://github.com/dan64/vs-deoldify/blob/main/test_images_ex/Image_10_test_ex.jpg) | 8.1 | 7.7 | 8.0 | **7.1** | 9.0 |
|[11](https://github.com/dan64/vs-deoldify/blob/main/test_images_ex/Image_11_test_ex.jpg) | **7.2** | 7.3 | 7.4 | 8.6 | 7.9 |
|[12](https://github.com/dan64/vs-deoldify/blob/main/test_images_ex/Image_12_test_ex.jpg) | 8.0 | 7.1 | 8.0 | **6.5** | 9.3 |
|[13](https://github.com/dan64/vs-deoldify/blob/main/test_images_ex/Image_13_test_ex.jpg) | 12.0 | **11.7** | 12.0 | 11.8 | 13.8 |
|[14](https://github.com/dan64/vs-deoldify/blob/main/test_images_ex/Image_14_test_ex.jpg) | **4.5** | 4.6 | 4.8 | 5.8 | 4.8 |
|[15](https://github.com/dan64/vs-deoldify/blob/main/test_images_ex/Image_15_test_ex.jpg) | 8.3 | **8.1** | 8.9 | 8.2 | 12.2 |
|[16](https://github.com/dan64/vs-deoldify/blob/main/test_images_ex/Image_16_test_ex.jpg) | 10.6 | 10.5 | 10.7 | 12.5 | **9.9** |
|[17](https://github.com/dan64/vs-deoldify/blob/main/test_images_ex/Image_17_test_ex.jpg) | **10.8** | 12.1 | 11.4 | 12.3 | 13.5 |
|[18](https://github.com/dan64/vs-deoldify/blob/main/test_images_ex/Image_18_test_ex.jpg) | 6.7 | 7.1 | **6.1** | 11.1 | 7.2 |
|[19](https://github.com/dan64/vs-deoldify/blob/main/test_images_ex/Image_19_test_ex.jpg) | **3.5** | 4.6 | 4.5 | 5.1 | 7.1 |
|[20](https://github.com/dan64/vs-deoldify/blob/main/test_images_ex/Image_20_test_ex.jpg) | 8.0 | 8.1 | 8.2 | 9.3 | **7.6** |
|[21](https://github.com/dan64/vs-deoldify/blob/main/test_images_ex/Image_21_test_ex.jpg) | 6.9 | **6.7** | 7.1 | 7.1 | 9.0 |
|[22](https://github.com/dan64/vs-deoldify/blob/main/test_images_ex/Image_22_test_ex.jpg) | 12.1 | 11.0 | **10.9** | 12.1 | 11.2 |
|[23](https://github.com/dan64/vs-deoldify/blob/main/test_images_ex/Image_23_test_ex.jpg) | 6.2 | 6.3 | **6.0** | 7.8 | 10.2 |
|**Avg(dE)** | **8.0** | **8.0** | **8.1** | **8.9** | **9.4** |

First of all, it should be noted that the individual models added (**DA** for _DeOldify_ and **DDs** for _DDColor_)  performed worse than the individual models tested in the previous analysis (**DS** for _DeOldify_ and **DD** for _DDColor_). Conversely all combinations of _DeOldify_ and _DDColor_ performed well.  Confirming the positive impact on the final result, already observed in the previous analysis, obtained by combining the 2 models. 

## Exemplar-based Models

As stated previously to stabilize further the colorized videos it is possible to use the frames colored by HAVC as reference frames (exemplar) as input to the supported exemplar-based models: [ColorMNet](https://github.com/yyang181/colormnet),  [Deep Exemplar based Video Colorization](https://github.com/zhangmozhe/Deep-Exemplar-based-Video-Colorization) and [DeepRemaster](https://github.com/satoshiiizuka/siggraphasia2019_remastering). 

In Hybrid the _Exemplar Models_ have their own panel, as shown in the following picture: 
![Hybrid DeepEx](https://github.com/dan64/vs-deoldify/blob/main/hybrid_setup/Model_DeepEx.JPG)   

For the ColorMNet models there are 2 implementations defined, by the field **Mode**:

- 'remote'  (has not memory frames limitation but it uses a remote process for the inference)
- 'local' (the inference is performed inside the Vapoursynth local thread but has memory limitation)

The field **Preset** control the render method and speed, allowed values are:

- 'Fast'  (faster but colors are more washed out)
- 'Medium' (colors are a little washed out)
- 'Slow' (slower but colors are a little more vivid)

The field **SC thresh** define the sensitivity for the scene detection (suggested value **0.1**, see [Miscellaneous Filters](https://amusementclub.github.io/doc3/plugins/misc.html)), while the field **SC min freq** allows to specify the minimum number of reference frames that have to be generated.

The  flag **Vivid** has 2 different meanings depending on the _Exemplar Model_ used:

- __ColorMNet__ (the frames memory is reset at every reference frame update)
- __DeepEx__, __DeepRemaster__ (given that the colors generated by the  inference are a little washed out , the saturation of colored frames will be increased by about 25%).

The field **Method** allows to specify the type of reference frames (RF) provided in input to the _Exemplar-based Models_, allowed values are:
- 0 = HAVC same as video (default)
- 1 = HAVC + RF same as video
- 2 = HAVC + RF different from video
- 3 = external RF same as video
- 4 = external RF different from video
- 5 = external ClipRef same as video
- 6 = external ClipRef different from video

It is possible to specify the directory containing the external reference frames by using the field **Ref FrameDir**. The frames must be named using the following format: _ref_nnnnnn.[png|jpg]_. For the methods 5 and 6 it is possible to pass a video clip as source for reference images.

Unfortunately all the Deep-Exemplar methods have the problem that are unable to properly colorize the new "features" (new elements not available in the reference frame) so that often these new elements are colored with implausible colors (see for an example: [New "features" are not properly colored](https://github.com/yyang181/NTIRE23-VIDEO-COLORIZATION/issues/10)). To try to fix this problem has been introduced the possibility to merge the frames propagated by DeepEx with the frames colored with DDColor and/or DeOldify. The merge is controlled by the field **Ref merge**, allowed values are:
- 0 = no merge
- 1 = reference frames are merged with low weight
- 2 = reference frames are merged with medium weight
- 3 = reference frames are merged with high weight

When the field **Ref merge** is set to a value greater than 0, the field **SC min freq** is set =1, to allows the merge for every frame (more details are provided in [HAVC User Guide](https://github.com/dan64/vs-deoldify/blob/main/documentation/HAVC%20User%20Guide.pdf)).   

Finally the flag **Reference frames only** can be used to export the reference frames generated with the method **HAVC** and defined by the parameters  **SC thresh** ,  **SC min freq** fields. 

## Coloring using Hybrid

As stated previously the simplest way to colorize images with the HAVC filter it to use [Hybrid](https://www.selur.de/downloads). To simplify the usage has been introduced standard Presets that automatically apply all the filter's settings. A set of parameters that are able to provide a satisfactory colorization are the following:

- **Speed:** medium or fast (_fast_ will increase the speed with a little decrease in color accuracy)
- **Color map:** none
- **Color tweaks:**  violet/red
- **Denoise:** light
- **Stabilize:** stable or morestable

then enable the _Exemplar Models_ check box and set
- **Method:** HAVC
- **SC thresh:** 0.10
- **SC SSIM thresh:** 0.0
- **SC min freq:** 15 (5 if is used the _local_ mode)  
- **normalize:** checked
- **Mode:** remote
- **Frames:** 0 
- **Preset:** medium (_slow_ will increase the color accuracy but the speed will decrease of 40%)
- **Vivid:** checked 

In the following picture are shown the suggested parameters: 

![Hybrid Preset](https://github.com/dan64/vs-deoldify/blob/main/hybrid_setup/Model_Presets.JPG)   

## Conclusions

In Summary **HAVC** is able to provide often a final colorized image that is better than the image obtained from the individual models, and can be considered an improvement respect to the current Models.  It is highly recommended to read the [HAVC User Guide](https://github.com/dan64/vs-deoldify/blob/main/documentation/HAVC%20User%20Guide.pdf) which provides useful tips on how to improve the colored movies.   

As a final consideration I would like to point out that the test results showed that the images coloring technology is mature enough to be used concretely both for coloring images and, thanks to **Hybrid**, videos.

## Acknowledgements

I would like to thank Selur, author of [Hybrid](https://www.selur.de/), for his wise advices and for having developed a gorgeous interface for this filter. Despite the large number of parameters and the complexity of managing them appropriately, the interface developed by Selur makes its use easy even for non-experts users.


  
