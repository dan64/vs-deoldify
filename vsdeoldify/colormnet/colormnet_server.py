"""
-------------------------------------------------------------------------------
Author: Dan64
Date: 2024-09-27
version:
LastEditors: Dan64
LastEditTime: 2024-11-02
-------------------------------------------------------------------------------
Description:
-------------------------------------------------------------------------------
ColorMNet frame server class for Vapoursynth.
"""
import os
from os import path
import warnings
import torch
import numpy as np
from PIL import Image
import threading
from xmlrpc.server import SimpleXMLRPCServer
from xmlrpc.server import SimpleXMLRPCRequestHandler

from vsdeoldify.colormnet import ColorMNetRender
from vsdeoldify.colormnet.colormnet_utils import *

package_dir = os.path.dirname(os.path.realpath(__file__))


class ColorMNetRPCServer:
    server_address: str = None
    server_port: int = None
    server: SimpleXMLRPCServer = None

    # Restrict to a particular path.
    class RequestHandler(SimpleXMLRPCRequestHandler):
        rpc_paths = ('/RPC2',)

    def __init__(self, server_address: str = '127.0.0.1', server_port: int = 0):
        self.server_address = server_address
        # Register an instance; all the methods of the instance are
        # published as XML-RPC methods.
        self.server = SimpleXMLRPCServer(addr=(server_address, server_port), allow_none=True,
                                         requestHandler=self.RequestHandler, use_builtin_types=True, logRequests=False)
        self.server_port = self.server.socket.getsockname()[1]
        self.server.register_introspection_functions()
        self.server.register_instance(self.ColorMNetService(), allow_dotted_names=True)

    def shutdown(self):
        self.server.shutdown()

    class ColorMNetService:
        render: ColorMNetRender = None

        def initialize(self, image_size: int = -1, vid_length: int = 1000, enable_resize: bool = False,
                       encode_mode: int = 0, propagate: bool = False, max_memory_frames: int = None,
                       reset_on_ref_update: bool = True):
            self.render = ColorMNetRender(image_size, vid_length, enable_resize, encode_mode, propagate,
                                          max_memory_frames, reset_on_ref_update=reset_on_ref_update,
                                          project_dir=package_dir)

        def SetRefImage(self, img_byte_array: bytes, frame_propagate: bool = False):
            img = byte_array_to_image(img_byte_array)
            if self.render is not None:
                self.render.set_ref_frame(img, frame_propagate)
            else:
                warnings.warn("ColorMNet Render is not initialized")

        def SetRefImageNone(self, frame_propagate: bool = False):
            img = None
            if self.render is not None:
                self.render.set_ref_frame(img, frame_propagate)
            else:
                warnings.warn("ColorMNet Render is not initialized")

        def IsInitialized(self) -> bool:
            return self.render is not None

        def ColorizeImage(self, img_byte_array: bytes, ti: int = None):
            img = byte_array_to_image(img_byte_array)
            if self.render is not None:
                img_colored = self.render.colorize_frame(ti, img)
                img_byte_array = image_to_byte_array(img_colored)
                return img_byte_array
            else:
                warnings.warn("ColorMNet Render is not initialized")
                return img_byte_array

    def start_server(self):
        warnings.warn("start ColorMNet server, listening on : " + str(self.server.server_address))
        # Run the server's main loop
        self.server.serve_forever()


class ColorMNetServer:
    _instance = None
    _initialized = False
    rpc_server: ColorMNetRPCServer = None
    rpc_thread: threading.Thread = None
    context: any = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, server_port: int = 0):
        if not self._initialized:
            try:
                server_address = '127.0.0.1'
                self.rpc_server = ColorMNetRPCServer(server_address, server_port)
                self.rpc_thread = threading.Thread(target=self.rpc_server.start_server, name="RPCServer-1", daemon=True)
                self._initialized = True
            except Exception as exe:
                warnings.warn("ERROR[" + str(type(exe)) + "]: " + str(exe))

    def run_server(self):
        if self.rpc_thread is None:
            return None
        if not self.rpc_thread.is_alive():
            self.rpc_thread.start()
        return self

    def get_port(self):
        return self.rpc_server.server_port

    def close_server(self):
        if self.rpc_thread.is_alive():
            warnings.warn("ColorMNet server is alive, stop it")
            self.rpc_server.shutdown()
            self.rpc_thread.join()
        warnings.warn("ColorMNet server closed")
