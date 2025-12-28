"""
-------------------------------------------------------------------------------
Author: Dan64
Date: 2025-10-04
version:
LastEditors: Dan64
LastEditTime: 2025-10-04
-------------------------------------------------------------------------------
Description:
-------------------------------------------------------------------------------
HAVC functions library.
"""
import os

# Hybrid paths for plugins used by HAVC
vsslib_dir: str = os.path.dirname(os.path.realpath(__file__))

support_dir: str = os.path.join(vsslib_dir, "..", "..", "..", "..", "..", "vsfilters", "Support")

# Path for ReduceFlicker: https://github.com/AmusementClub/ReduceFlicker
ReduceFlicker_dir: str = os.path.join(vsslib_dir, "..", "..", "..", "..", "..", "vsfilters", "FrameFilter", "ReduceFlicker")

# Path for Retinex: https://github.com/HomeOfVapourSynthEvolution/VapourSynth-Retinex
Retinex_dir: str = os.path.join(vsslib_dir, "..", "..", "..", "..", "..", "vsfilters", "ColorFilter", "Retinex")

# Path for SCDetect: https://github.com/vapoursynth/vs-miscfilters-obsolete
MiscFilter_dir: str = os.path.join(vsslib_dir, "..", "..", "..", "..", "..", "vsfilters", "MiscFilter", "MiscFilters")

# Path for LSMASHSource: https://github.com/AmusementClub/ReduceFlicker
LSMASHSource_dir: str = os.path.join(vsslib_dir, "..", "..", "..", "..", "..", "vsfilters", "SourceFilter", "LSmashSource")

# Path for TimeCube: https://github.com/sekrit-twc/timecube
TimeCube_dir: str = os.path.join(vsslib_dir, "..", "..", "..", "..", "..", "vsfilters", "ColorFilter", "TimeCube")
