"""
-------------------------------------------------------------------------------
Author: Dan64
Date: 2025-09-28
version:
LastEditors: Dan64
LastEditTime: 2025-09-28
-------------------------------------------------------------------------------
Description:
-------------------------------------------------------------------------------
ColorMNet frame client class for Vapoursynth.
"""
import os
from PIL import Image
import warnings
import xmlrpc.client
from vsdeoldify.colormnet2.colormnet2_utils import *


class ColorMNetClient2:
    _instance = None
    _initialized = False
    server_address: str = None
    server_port: int = None
    server: xmlrpc.client.ServerProxy = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, image_size: int = -1, vid_length: int = 1000, enable_resize: bool = False,
                 encode_mode: int = 0, propagate: bool = False, max_memory_frames: int = None,
                 reset_on_ref_update: bool = True, server_port: int = None):
        if not self._initialized:
            server_address = '127.0.0.1'
            if server_port is None:
                warnings.warn("ERROR: ColorMNetClient() server port is None")
                return
            self.server_address = server_address
            self.server_port = server_port
            # Connect to a RPC instance; all the methods of the instance are
            # published as XML-RPC methods.
            self.uri = f"http://{server_address}:{server_port}"
            try:
                self.server = xmlrpc.client.ServerProxy(uri=self.uri, allow_none=True, use_builtin_types=True)
                self.server.initialize(image_size, vid_length, enable_resize, encode_mode, propagate,
                                       max_memory_frames, reset_on_ref_update)
                self._initialized = True
            except Exception as exe:
                warnings.warn("ERROR[" + str(type(exe)) + "]: " + str(exe))

    def is_initialized(self) -> bool:
        return self.server.IsInitialized()

    def get_frame_count(self) -> int:
        return self.server.GetFrameCount()

    def set_ref_frame(self, frame_ref: Image = None, frame_propagate: bool = False):
        if frame_ref is None:
            self.server.SetRefImageNone(frame_propagate)
        else:
            frame_bytes = image_to_byte_array(frame_ref)
            self.server.SetRefImage(frame_bytes, frame_propagate)

    def colorize_frame(self, ti: int = None, frame_i: Image = None) -> Image:
        if frame_i is not None:
            img_bytes_i = image_to_byte_array(frame_i)
            frame_bytes = self.server.ColorizeImage(img_bytes_i, ti)
            return byte_array_to_image(frame_bytes)
        else:
            return None
