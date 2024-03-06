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

Taking inspiration from the article published on Habr: [Mode on: Comparing the two best colorization AI's](https://habr.com/en/companies/ruvds/articles/568426/). I decided to use it to get the refence images and the images obtained using the [ColTran](https://github.com/google-research/google-research/tree/master/coltran) model, to extend the analysis with the models implemented in the **DDeoldify** filter.

The added models are:

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

To compare the models I decided to use a metric being able to consider the _perceptual non-uniformities_ in the evaluation of color difference between images. These non-uniformities are important because the human eye is more sensitive to certain colors than others.  Over time, The International Commission on Illumination (**CIE**) has proposed increasingly advanced measurement models to measure the color distance taking into account the _human color perception_, that they called **dE**. One of the most advanced is the [CIEDE2000](https://en.wikipedia.org/wiki/Color_difference#CIEDE2000) method, that I decided to use as _color similarity metric_ to compare the models. The final results are shown in the table below (test image can be seen by clicking on the test number)

| Test # | D+D | DD | DS | DV  | T241 |
|------|------|-----|-----|-----|-------|
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




       
The calculation of **dE** with the  **CIEDE2000** model was obtained by leveraging on the computational code available in [ColorMine](https://github.com/MasterPieceCode/Mozaic/tree/master/ColorMine).

As it is possible to see the model that performed better is the **D+D** model (which I called _DDelodify_ because is using both _Deoldify_ and _DDColor_). This model was the best model in 10 tests out of 23. Also the **DD** model performed well but there were situations where the **DD** model provided quite bad colorized images like in [Test #23](https://github.com/dan64/vs-deoldify/blob/main/test_images/Image_23_test.jpg) and the combination with the Deoldify allowed to significantly improve the final image. In effect the average distance of **DD** was 8.3 while for **DV** was 9.5, given that the 2 models were weighted at 50%, if the images were positively correlated a value 9 would have been expected, instead the average distance measured for **D+D** was 8.3, this implies that the 2 models were able to compensate each other. 
Conversely, the **T241** was the model that performed worse with the greatest average difference in colors. Finally, the quality of Deoldify models was similar, being **DS** slightly better than **DV** (as expected).

In Summary **DDeoldify** is able to provide often a final colorized image that is better than the image obtained from the individual models, and can be considered an improvement respect to the current Models.   

As a final consideration I would like to point out that the test results showed that the images coloring technology is mature enough to be used concretely both for coloring images and, thanks to Hybrid, videos.




  