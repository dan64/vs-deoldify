# DDeoldify
A Deep Learning based Vaoursynth filter for colorizing and restoring old images and video, based on [DeOldify](https://github.com/jantic/DeOldify)
and  [DDColor](https://github.com/HolyWu/vs-ddcolor) 

The Vapoursynth filter version has the advantage of coloring the images directly in memory, without the need to use the filesystem to store the video frames. 

This filter is able to combine the results provided by DeOldify and DDColor, which are some of the best models available for coloring pictures, providing often a final colorized image that is better than the image obtained from the individual models.  


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
