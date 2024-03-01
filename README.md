# Deoldify
A Deep Learning based Vaoursynth filter for colorizing and restoring old images and video, based on [https://github.com/jantic/DeOldify](https://github.com/jantic/DeOldify).

The Vapoursynth filter version has the advantage of coloring the images directly in memory, without the need to use the filesystem to store the video frames.


## Dependencies
- [PyTorch](https://pytorch.org/get-started) 2.1.1 or later
- [VapourSynth](http://www.vapoursynth.com/) R62 or later


## Installation
```
pip install -U vsdeoldify
```


## Models Download
The models are not installed with the package, they must be downloaded from the Deoldify website at: [completed-generator-weights](https://github.com/jantic/DeOldify#completed-generator-weights).

The models to download are:

- ColorizeVideo_gen.pth
- ColorizeStable_gen.pth
- ColorizeArtistic_gen.pth

The model files have to be copied in the model dir, usually located in:

.\Lib\site-packages\vsdeoldify\models


At the first usage it is possible that are automatically downloaded by torch the neural networks: resnet101 and resnet34. 

It is possible specify the destination directory of networks used by torch, by using the function parameter **torch\_hub\_dir**, if this parameter is set to **None**, the files will be downloaded in the torch cache dir, more details are available at: [caching-logic ](https://pytorch.org/docs/stable/hub.html#caching-logic).


## Usage
```python
from vsdeoldify import ddeoldify

ret = ddeoldify(clip)
```

See `__init__.py` for the description of the parameters.
