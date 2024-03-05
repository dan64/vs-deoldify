# DDeoldify
A Deep Learning based Vaoursynth filter for colorizing and restoring old images and video, based on [DeOldify](https://github.com/jantic/DeOldify)
and  [DDColor](https://github.com/HolyWu/vs-ddcolor) 

The Vapoursynth filter version has the advantage of coloring the images directly in memory, without the need to use the filesystem to store the video frames. 

This filter is able to combine the results provided by DeOldify and DDColor, which are some of the best models available for coloring pictures, providing often a final colorized image that is better than the image obtained from the individual models.  

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

The _model files_ have to be copied in the **models** directory (which must be created manually the first time), usually located in:

.\Lib\site-packages\vsdeoldify\models


At the first usage it is possible that are automatically downloaded by torch the neural networks: resnet101 and resnet34. 

It is possible specify the destination directory of networks used by torch, by using the function parameter **torch\_hub\_dir**, if this parameter is set to **None**, the files will be downloaded in the torch cache dir, more details are available at: [caching-logic ](https://pytorch.org/docs/stable/hub.html#caching-logic).

The models used by DDColor can be installed with the command

```
python -m vsddcolor
```


## Usage
```python
from vsdeoldify import ddeoldify
# DeOldify only model
clip = ddeoldify(clip)
# DeOldify with DDColor weighed at 50%
clip = ddeoldify(clip, dd_weight=0.5)

```

See `__init__.py` for the description of the parameters.

## Comparison of Models ##

Taking inspiration from the article published on Habr: [Mode on: Comparing the two best colorization AI's](https://habr.com/en/companies/ruvds/articles/568426/). I decide to use it to get the refence images and the images obtained using the [ColTran](https://github.com/google-research/google-research/tree/master/coltran) model, to extend the analysis to include the models implemented in this filter.

The added Models are:

**D+D**: Deoldify (with model _Video_ & render_factor = 23) + DDColor (with model _Artistic_ and input_size = 3)
![Hybrid D+D](https://github.com/dan64/vs-deoldify/blob/main/hybrid_setup/Model_D%2BD.JPG)  

**DD**:  DDColor (with model _Artistic_ and input_size = 384)
![Hybrid_DD](https://github.com/dan64/vs-deoldify/blob/main/hybrid_setup/Model_DD.JPG)

**DS**: Deoldify (with model _Stable_ & render_factor = 30)
![Hybrid D+D](https://github.com/dan64/vs-deoldify/blob/main/hybrid_setup/Model_DS.JPG)  

**DV**: Deoldify (with model _Video_ & render_factor = 23)
![Hybrid D+D](https://github.com/dan64/vs-deoldify/blob/main/hybrid_setup/Model_DV.JPG)  

**T241**:  ColTran + TensorFlow 2.4.1 model as shown in [Habr](https://habr.com/en/companies/ruvds/articles/568426/)

**Comparison Methodology**

To compare the models I decided to use a metric being able to consider the _perceptual non-uniformities_ in the evaluation of color difference between images. These non-uniformities are important because the human eye is more sensitive to certain colors than others.  Over time, The International Commission on Illumination (**CIE**) has proposed increasingly advanced measurement models. One of the most advance is the [CIEDE2000](https://en.wikipedia.org/wiki/Color_difference#CIEDE2000) that I decided to use to compare the models. The final results are shown in the table below (the test images are available in the folder [test_images](https://github.com/dan64/vs-deoldify/tree/main/test_images))

![](https://github.com/dan64/vs-deoldify/blob/main/test_images/Comparison_Results_CIEDE2000.PNG)

As it is possible to see the model that performed better is the **D+D** model (which I called _DDelodify_ because is using both _Deoldify_ and _DDColor_). This model was the best model in 10 tests out of 23. Also the **DD** model performed well but there were situation where the **DD** model provided quite bad colorized images like in [Test #23](https://github.com/dan64/vs-deoldify/blob/main/test_images/Image_23_test.jpg) and the combination with the Deoldify allowed to significantly improve the final image. The **T241** was the model that performed worse with the greatest average difference in colors. Conversely, the quality of Deoldify models was similar, being **DS** slightly better than **DV** (as expected).

In Summary **DDeoldify** is able to provide often a final colorized image that is better than the image obtained from the individual models, and can be considered an improvement respect to the current Models.   

As a final consideration I would like to point out that the test results showed that the images coloring technology is mature enough to be used concretely both for coloring images and, thanks to Hybrid, videos.




  