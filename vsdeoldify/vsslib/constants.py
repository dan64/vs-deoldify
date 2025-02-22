"""
-------------------------------------------------------------------------------
Author: Dan64
Date: 2024-11-20
version:
LastEditors: Dan64
LastEditTime: 2025-02-21
-------------------------------------------------------------------------------
Description:
-------------------------------------------------------------------------------
Constants for vs-deoldify functions.
"""

DEF_THT_WHITE: float = 0.88
DEF_THT_BLACK: float = 0.12
DEF_THT_BLACK_FREQ: float = 0.14
DEF_THT_BLACK_MIN: float = 0.19
DEF_THT_WHITE_MIN: float = 0.81
DEF_ADAPTIVE_RATIO_LO: float = 1.02
DEF_ADAPTIVE_RATIO_MED: float = 1.12
DEF_ADAPTIVE_RATIO_HI: float = 1.20
DEF_ADAPTIVE_RATIO_RF: float = 2.0
DEF_ADAPTIVE_RATIO_VHI: float = 15.0
DEF_SSIM_SCORE_EQUAL: float = 0.69
DEF_HIST_SCORE_EQUAL: float = 0.70
DEF_HIST_SCORE_HIGH: float = 0.90
DEF_MERGE_LOW_WEIGHT: float = 0.20
DEF_EXPORT_FORMAT: str = 'jpg'
DEF_JPG_QUALITY: int = 95
DEF_THRESHOLD: float = 0.10
DEF_MIN_FREQ: int = 10               # MIN reference frames frequency for deep-remaster
DEF_MAX_FREQ: int = 15               # MAX reference frames frequency for deep-remaster
DEF_SC_MIN_DISTANCE: int = 15
DEF_MAX_MEMORY_FRAMES: int = 10000   # theoretically MAX_MEMORY must be < 95000
DEF_MAX_RF_FRAMES: int = 200         # MAX number of reference frames to use for deep-remster
DEF_NUM_RF_FRAMES: int = 10          # default number of reference frames to use for deep-remster
DEF_MIN_RF_FRAMES: int = 4           # MIN number of reference frames to use for deep-remster
DEF_MAX_BUFFER_SIZE: int = 500       # number of frames to scan for searching RF for deep-remster
DEF_MAX_XREF_BUFFER: int = 500       # number of frames to scan for searching RF for colormnet
DEF_MAX_XRF_FRAMES: int = 250        # MAX number of reference frames to load at start for colormnet
DEF_MAX_XREF_WINDOW: int = 20        # number of forward reference frames for colormnet
DEF_NUM_XRF_FRAMES: int = 30         # default number of reference frames to load at start for colormnet
DEF_MIN_XRF_FRAMES: int = 4          # MIN number of reference frames to load at start for colormnet
DEF_FUTURE_FRAME_WEIGHT: float = 0.5
DEF_BATCH_SIZE: int = 2
DEF_VIVID_HUE_LOW: float = 3.0
DEF_VIVID_SAT_HIGH: float = 1.30
DEF_VIVID_HUE_HIGH: float = 5.0
DEF_VIVID_SAT_LOW: float = 1.15
DEF_MIN_COLOR_ALPHA: float = 1.0
DEF_MAX_COLOR_ALPHA: float = 10.0




