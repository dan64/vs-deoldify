# Deoldify
A Deep Learning based project for colorizing and restoring old images and video, based on https://github.com/jantic/DeOldify


## Dependencies
- [PyTorch](https://pytorch.org/get-started) 2.1.1 or later
- [VapourSynth](http://www.vapoursynth.com/) R62 or later


## Installation
```
pip install -U vsdeoldify
```


## Usage
```python
from vsdeoldify import ddeoldify

ret = ddeoldify(clip)
```

See `__init__.py` for the description of the parameters.
