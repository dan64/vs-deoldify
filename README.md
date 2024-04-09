# DDeoldify
A Deep Learning based Vaoursynth filter for colorizing and restoring old images and video, based on [DeOldify](https://github.com/jantic/DeOldify)
and  [DDColor](https://github.com/HolyWu/vs-ddcolor) 

The Vapoursynth filter version has the advantage of coloring the images directly in memory, without the need to use the filesystem to store the video frames. 

This filter is able to combine the results provided by DeOldify and DDColor, which are some of the best models available for coloring pictures, providing often a final colorized image that is better than the image obtained from the individual models.  But the main strength of this filter is the addition of specialized filters to improve the quality of videos obtained by using these color models. 

## Quick Start

This filter is distributed with the torch package provided with the **Hybrid Windows Addons**. To use it on Desktop (Windows, Linux) it is necessary install [Hybrid](https://www.selur.de/downloads). **Hybrid** is a Qt-based frontend for other tools (including this filter) which can convert most input formats to common audio & video formats and containers. It represent the  easiest way to colorize images with [DeOldify](https://github.com/jantic/DeOldify) using [VapourSynth](https://www.vapoursynth.com/).      


## Dependencies
- [PyTorch](https://pytorch.org/get-started) 2.1.1 or later
- [VapourSynth](http://www.vapoursynth.com/) R62 or later

## Installation
```
pip install vsdeoldify-x.x.x-py3-none-any.whl
```

## Models Download
The models are not installed with the package, they must be downloaded from the Deoldify website at: [completed-generator-weights](https://github.com/jantic/DeOldify#completed-generator-weights).

The models to download are:

- ColorizeVideo_gen.pth
- ColorizeStable_gen.pth
- ColorizeArtistic_gen.pth

The _model files_ have to be copied in the **models** directory usually located in:

.\Lib\site-packages\vsdeoldify\models

At the first usage it is possible that are automatically downloaded by torch the neural networks: resnet101 and resnet34. 

So don't be worried if at the first usage the filter will be very slow to start, at the initialization are loaded almost all the _Fastai_ and _PyTorch_ modules and the resnet networks.

It is possible specify the destination directory of networks used by torch, by using the function parameter **torch\_hub\_dir**, if this parameter is set to **None**, the files will be downloaded in the _torch's cache_ dir, more details are available at: [caching-logic ](https://pytorch.org/docs/stable/hub.html#caching-logic).

The models used by **DDColor** can be installed with the command

```
python -m vsddcolor
```


## Usage
```python
from vsdeoldify import ddeoldify
# DeOldify  with DDColor weighed at 40%
clip = ddeoldify(clip)
# DeOldify only model
clip = ddeoldify(clip, method=0)
# DDColor only model
clip = ddeoldify(clip, method=1)

# To apply video color stabilization filters for ddeoldify
from vsdeoldify import ddeoldify_stabilizer
clip = ddeoldify_stabilizer(clip, dark=True, smooth=True, stab=True)
```

See `__init__.py` for the description of the parameters.

**NOTE**: In the _DDColor_ version included with **DDeoldify** the parameter _input_size_ has changed name in _render_factor_ because were changed the range of values of this parameter to be equivalent to _render_factor_ in _Deoldify_, the relationship between these 2 parameters is the following:

```
input_size = render_factor * 16
``` 

## Filter Usage

The filter was developed having in mind to use it mainly to colorize movies. Both Deoldify and DDcolor are good models for coloring pictures (see the _Comparison of Models_). But when are used for coloring movies they are introducing artifacts that usually are not  noticeable in the images.  Especially in dark scenes both Deoldify and DDcolor are not able to understand what it is the dark area and what color to give it, they often decide to color these dark areas with blue, then in the next frame this area could become red and then in the next frame return to blue, introducing a flashing psychedelic effect when all the frames are put in a movie.
To try to solve this problem has been developed _pre-_ and _post-_ process filters.  It is possible to see them in the Hybrid screenshot below.

![Hybrid Coloring page](https://github.com/dan64/vs-deoldify/blob/main/hybrid_setup/Model_D%2BD_filters.JPG)  

The main filters introduced are:

**Chroma Smoothing** : This filter allows to to reduce the _vibrancy_ of colors assigned by Deoldify/DDcolor by using the parameters _de-saturation_ and _de-vibrancy_ (the effect on _vibrancy_ will be visible only if the option **chroma resize** is enabled, otherwise this parameter has effect on the _luminosity_). The area impacted by the filter is defined by the thresholds dark/white. All the pixels with luma below the dark threshold will be impacted by the filter, while the pixels above the white threshold will be left untouched. All the pixels in the middle will be gradually impacted depending on the luma value.

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
specify the range of hue colors: 290-360, because "rose" is hue wheel name that correspond to the range:330-360.

It is possible to specify more ranges by using the comma "," separator.

When the de-saturation information is not already available in the filter's parameters, it necessary to use the following syntax:

```
chroma_adjustment = "chroma_range|sat,weight"
``` 

in this case it is necessary to specify also the de-saturation parameter "sat" and the blending parameter "weight".

for example with this assignment: 

```
chroma_range = "300:330|0.4,0.2"
``` 

the hue colors in the range 300-340 will be de-saturated by the amount 0.4 and the final frame will be blended by applying a 20% de-saturation of 0.4 an all the pixels.  


 
### Merging the models

As explained previously, this filter is able to combine the results provided by DeOldify and DDColor, to perform this combination has been implemented 6 methods:

0. _DeOldify_ only coloring model.

1. _DDColor_ only color model.

2. _Simple Merge_: the frames are combined using a _weighted merge_, where the parameter _merge_weight_ represent the weight assigned to the frames provided by the DDcolor model. 

3. _Constrained Chroma Merge_:  given that the colors provided by Deoldify's _Video_ model are more conservative and stable than the colors obtained with DDcolor. The frames are combined by assigning a limit to the amount of difference in chroma values between Deoldify and DDcolor. This limit is defined by the parameter _threshold_. The limit is applied to the frame converted to "YUV". For example when threshold=0.1, the chroma    values "U","V" of DDcolor frame will be constrained to have an absolute percentage difference respect to "U","V" provided by Deoldify not higher than 10%.  If _merge_weight_ is < 1.0, the chroma limited DDColor frames will be will be merged again with the frames of Deoldify using the _Simple Merge_.

 4. _Luma Masked Merge_: the behaviour is similar to the method _Adaptive Luma Merge_. With this method the frames are combined using a _masked merge_. The pixels of DDColor's frame with luma < _luma_limit_  will be filled with the (de-saturated) pixels of Deoldify, while the pixels above the _white_limit_ threshold will be left untouched. All the pixels in the middle will be gradually replaced depending on the luma value. If the parameter  _merge_weight_ is < 1.0, the resulting masked frames will be merged again with the non de-saturated frames of Deoldify using the _Simple Merge_.

5. _Adaptive Luma Merge_: given that the DDcolor perfomance is quite bad on dark scenes, with this method the images are combined by decreasing the weight assigned to DDcolor frames when the luma is below the _luma_threshold_. For example with: luma_threshold = 0.6 and alpha = 1, the weight assigned to DDcolor frames will start to decrease linearly when the luma < 60% till _min_weight_. For _alpha_=2, the weight begins to decrease quadratically.      

The merging methods 2-5 are levereging on the fact that usually the Deoldify _Video_ model provides frames which are more stable, this feature is exploited to stabilize also DDColor. The methods 3 and 4 are similar to _Simple Merge_, but before the merge with _DeOldify_ the _DDColor_ frame is limited in the chroma changes (method 3) or limited based on the luma (method 4). The method 5 is a _Simple Merge_ where the weight decrease with luma. 

## Comparison of Models

Taking inspiration from the article published on Habr: [Mode on: Comparing the two best colorization AI's](https://habr.com/en/companies/ruvds/articles/568426/). I decided to use it to get the refence images and the images obtained using the [ColTran](https://github.com/google-research/google-research/tree/master/coltran) model, to extend the analysis with the models implemented in the **DDeoldify** filter.

The added models are:

**D+D**: Deoldify (with model _Video_ & render_factor = 23) + DDColor (with model _Artistic_ and render_factor = 24)
![Hybrid D+D](https://github.com/dan64/vs-deoldify/blob/main/hybrid_setup/Model_D%2BD.JPG)  

**DD**:  DDColor (with model _Artistic_ and input_size = 384)
![Hybrid_DD](https://github.com/dan64/vs-deoldify/blob/main/hybrid_setup/Model_DD.JPG)

**DS**: Deoldify (with model _Stable_ & render_factor = 30)
![Hybrid D+D](https://github.com/dan64/vs-deoldify/blob/main/hybrid_setup/Model_DS.JPG)  

**DV**: Deoldify (with model _Video_ & render_factor = 23)
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

As it is possible to see the model that performed better is the **D+D** model (which I called _DDelodify_ because is using both _Deoldify_ and _DDColor_). This model was the best model in 10 tests out of 23. Also the **DD** model performed well but there were situations where the **DD** model provided quite bad colorized images like in [Test #23](https://github.com/dan64/vs-deoldify/blob/main/test_images/Image_23_test.jpg) and the combination with the Deoldify allowed to significantly improve the final image. In effect the average distance of **DD** was **8.5** while for **DV** was **9.5**, given that the 2 models were weighted at 50%, if the images were positively correlated a value **9.0** would have been expected, instead the average distance measured for **D+D** was **8.3**, this implies that the 2 models were able to compensate each other. 
Conversely, the **T241** was the model that performed worse with the greatest average difference in colors. Finally, the quality of Deoldify models was similar, being **DS** slightly better than **DV** (as expected).

###  Tests Set #2

Given the goodness of  **CIEDE2000** method to provide a reliable estimate of _human color perception_, I decided to provide an additional set of tests including some of the cases not considered previously.

The models added are:

**DA**: Deoldify (with model _Artistic_ & render_factor = 30) 

**DDs**: DDColor (with model _ModelScope_ and input_size = 384)

**DS+DD**: Deoldify (with model _Stable_ & render_factor = 30) + DDColor (with model _Artistic_ and render_factor  = 24)

**DA+DDs**: Deoldify (with model _Artistic_ & render_factor = 30) + DDColor (with model _ModelScope_ and render_factor  = 24)

**DA+DD**: Deoldify (with model _Artistic_ & render_factor = 30) + DDColor (with model _Artistic_ and render_factor  = 24)

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

First of all, it should be noted that the individual models added (**DA** for _Deoldify_ and **DDs** for _DDColor_)  performed worse than the individual models tested in the previous analysis (**DS** for _Deoldify_ and **DD** for _DDColor_). Conversely all combinations of _Deoldify_ and _DDColor_ performed well.  Confirming the positive impact on the final result, already observed in the previous analysis, obtained by combining the 2 models. 

## Conclusions

In Summary **DDeoldify** is able to provide often a final colorized image that is better than the image obtained from the individual models, and can be considered an improvement respect to the current Models.  The suggested configuration for _video encoding_ is: 

* **D+D**: Deoldify (with model _Video_ & render_factor = 24) + DDColor (with model _Artistic_ and render_factor = 24)

willing to accept a decrease in encoding speed of about 40% it is possible to improve _a little_ the colorization process by using the configuration:

* **DS+DD**: Deoldify (with model _Video_ & render_factor = 30) + DDColor (with model _Artistic_ and render_factor = 30)

It is also suggested to enable the _DDColor Tweaks_  (to apply the dynamic gamma correction) and the post-process filters: _Chroma Smoothing_ and _Chroma Stabilization_. Unfortunately is not possible provide a _one size fits-all solution_ and the filter parameters need to be adjusted depending on the type of video to be colored.  

As a final consideration I would like to point out that the test results showed that the images coloring technology is mature enough to be used concretely both for coloring images and, thanks to **Hybrid**, videos.

## Acknowledgements

I would like to thank Selur, author of [Hybrid](https://www.selur.de/), for his wise advices and for having developed a gorgeous interface for this filter. Despite the large number of parameters and the complexity of managing them appropriately, the interface developed by Selur makes its use easy even for non-experts users.


  