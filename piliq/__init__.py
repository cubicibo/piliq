#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MIT License

Copyright (c) 2023 cubicibo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import os
import sys
import ctypes
import logging

from typing import Union, Optional
from functools import wraps

from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from numpy import typing as npt
import numpy as np

def get_logger(name: str = 'piliq',
               level: int = logging.INFO,
               format_str: str = ' %(name)s: %(levelname).5s : %(message)s') -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(format_str.format(name))
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

_logger = get_logger()

@dataclass
class RGBA:
    r: np.uint8
    g: np.uint8
    b: np.uint8
    a: np.uint8

    def to_tuple(self) -> tuple[int, int, int, int]:
        return (self.r, self.g, self.b, self.a)

    def __bytes__(self) -> bytes:
        return bytes(self.to_tuple())

    @classmethod
    def from_bytes(cls, rgba_bytestring: bytes) -> 'RGBA':
        return cls(*list(map(ord, rgba_bytestring[:4])))

class PILIQ:
    _lib = None
    def __init__(self) -> None:
        assert __class__._lib is not None, "Library must be loaded in the class before instantiating objects."
        self._attr = self._lib.liq_attr_create()
        _logger.debug("Created liq attr handle.")

    def __del__(self) -> None:
        if self._attr:
            self._lib.liq_attr_destroy(self._attr)
            _logger.debug("Destroyed liq attr handle.")

    def __ensure_liq(liq_call):
        @wraps(liq_call)
        def _ensure_liq_f(self, *args, **kwargs):
            err_msg = "libimagequant library not loaded."
            assert self._lib is not None, err_msg
            return liq_call(self, *args, **kwargs)
        return _ensure_liq_f

    @classmethod
    def set_lib(cls, lib: Union[str, Path, ctypes.CDLL]) -> None:
        if lib is not None:
            if not isinstance(lib, ctypes.CDLL):
                lib = ctypes.CDLL(lib)
            assert cls._lib is None
            cls._write_interface(lib)
            cls._lib = lib
        ####

    @classmethod
    def is_ready(cls) -> bool:
        try:
            return (cls.get_version() > 0)
        except (OSError, AssertionError):
            return False

    @__ensure_liq
    def quantize(self, img: Image.Image, colors: int = 255) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
        assert img.mode == 'RGBA'
        assert 0 < colors <= 256
        self._lib.liq_set_max_colors(self._attr, colors)
        assert self._lib.liq_get_max_colors(self._attr) == colors

        img_array = np.ascontiguousarray(img, np.uint8)
        liq_img = self._lib.liq_image_create_rgba(self._attr, img_array.ctypes.data_as(ctypes.POINTER(ctypes.c_void_p)), img.width, img.height, 0)

        liq_res = ctypes.c_void_p()
        retval = self._lib.liq_image_quantize(liq_img, self._attr, ctypes.pointer(liq_res))
        if retval != 0:
            print("Error quantize")
            return None, None

        palette = self._lib.liq_get_palette(liq_res).contents.to_numpy()
        img_quantized = np.zeros((img.height, img.width), np.uint8, 'C')

        retval = self._lib.liq_write_remapped_image(liq_res, liq_img, img_quantized.ctypes.data_as(ctypes.POINTER(ctypes.c_void_p)), img_quantized.size)
        if retval != 0:
            print("Error getting back quant img")
            return None, None

        self._lib.liq_result_destroy(liq_res)
        self._lib.liq_image_destroy(liq_img)
        return palette, img_quantized
    ####

    @classmethod
    @__ensure_liq
    def get_version(cls) -> int:
        return cls._lib.liq_version()

    @staticmethod
    def _find_library() -> Optional[ctypes.CDLL]:
        LIB_NAME = 'libimagequant'
        if os.name == 'nt':
            import platform
            def get_dll_arch(fpath: Path) -> bool:
                # https://stackoverflow.com/a/65586112
                lut_arch = {332: 'I386', 512: 'IA64', 34404: 'AMD64', 452: 'ARM', 43620: 'AARCH64'}
                import struct
                with open(fpath, 'rb') as f:
                    assert f.read(2) == b'MZ'
                    f.seek(60)
                    f.seek(struct.unpack('<L', f.read(4))[0] + 4)
                    return lut_arch.get(struct.unpack('<H', f.read(2))[0], None)
                return None

            this_dir = Path(__file__).parent.absolute()
            lib_path = this_dir.joinpath('lib', LIB_NAME+'.dll')
            if lib_path.exists() and get_dll_arch(lib_path) == platform.machine():
                _logger.debug(f"Found embedded {lib_path} in folder, loading as DLL.")
                return ctypes.CDLL(str(lib_path))

        elif os.name == 'posix':
            import subprocess
            retry_linux = False

            ret = subprocess.run(['brew', '--prefix', LIB_NAME], capture_output=True)
            if sys.platform.lower() != 'darwin' and (ret.stderr != b'' or ret.stdout == b''):
                retry_linux = True
            elif ret.stdout != b'':
                if (lib_file := Path(str(ret.stdout, 'utf-8').strip()).joinpath('lib', LIB_NAME+'.dylib')).exists():
                    _logger.debug(f"Found {lib_file}, loading as dylib.")
                    return ctypes.CDLL(lib_file)

            if retry_linux:
                ret = subprocess.run(['dpkg', '-L', LIB_NAME+'-dev'], capture_output=True)
                for fpath in map(lambda bpath: Path(str(bpath, 'utf-8')), ret.stdout.split(b'\n')):
                    if str(fpath).endswith('.so') and fpath.exists():
                        _logger.debug(f"Found {fpath}, loading as shared object.")
                        return ctypes.CDLL(fpath)
        else:
            _logger.warning(f"Unknown OS, {LIB_NAME} has to be loaded manually.")
            return None
        _logger.warning(f"Failed to bind to {LIB_NAME}. Library must be set manually.")
        return None

    @staticmethod
    def _write_interface(lib: ctypes.CDLL) -> None:
        class LIQColor(ctypes.Structure):
            _fields_ = [("r", ctypes.c_char), ("g", ctypes.c_char), ("b", ctypes.c_char), ("a", ctypes.c_char)]

            def to_tuple(self) -> tuple[int, int, int, int]:
                return ord(self.r), ord(self.g), ord(self.b), ord(self.a)

            def to_rgba(self) -> RGBA:
                return RGBA(*self.to_tuple())

        class LIQPalette(ctypes.Structure):
            _fields_ = [("count", ctypes.c_uint), ("entries", LIQColor * 256)]

            def to_numpy(self) -> npt.NDArray[np.uint8]:
                return np.array([self.entries[k].to_tuple() for k in range(0, self.count)], dtype=np.uint8)

            def to_dict(self) -> dict[int, RGBA]:
                return {k: self.entries[k].to_rgba() for k in range(self.count)}

        liq_attr = ctypes.c_void_p
        liq_result = ctypes.c_void_p
        liq_image = ctypes.c_void_p

        lib.liq_version.argtype = (None,)
        lib.liq_version.restype = ctypes.c_uint

        #attr
        lib.liq_attr_create.argtype = (None,)
        lib.liq_attr_create.restype = ctypes.POINTER(liq_attr)
        lib.liq_attr_destroy.argtype = (ctypes.POINTER(liq_attr),)
        lib.liq_attr_destroy.restype = None

        #quantize opts
        lib.liq_set_max_colors.argtype = (ctypes.POINTER(liq_attr), ctypes.c_int)
        lib.liq_set_max_colors.restype = ctypes.c_int
        lib.liq_get_max_colors.argtype = (ctypes.POINTER(liq_attr),)
        lib.liq_get_max_colors.restype = ctypes.c_int

        lib.liq_image_create_rgba.argtype = (ctypes.POINTER(liq_attr), ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_double)
        lib.liq_image_create_rgba.restype = ctypes.POINTER(liq_image)

        lib.liq_image_quantize.argtype = (ctypes.POINTER(liq_image), ctypes.POINTER(liq_attr), ctypes.POINTER(ctypes.POINTER(liq_result)))
        lib.liq_image_quantize.restype = ctypes.c_int

        lib.liq_get_palette.argtype = (ctypes.POINTER(liq_result),)
        lib.liq_get_palette.restype = ctypes.POINTER(LIQPalette)

        lib.liq_write_remapped_image.argtype = (ctypes.POINTER(liq_result), ctypes.POINTER(liq_image), ctypes.c_void_p, ctypes.c_size_t)
        lib.liq_write_remapped_image.restype = ctypes.c_int

        lib.liq_image_destroy.argtype = (ctypes.POINTER(liq_image),)
        lib.liq_image_destroy.restype = None

        lib.liq_result_destroy.argtype = (ctypes.POINTER(liq_result),)
        lib.liq_result_destroy.restype = None

        _logger.debug(f"Loaded libimagequant version: '{lib.liq_version()}'.")
    ####
####

# Try to autoload the library by looking at the right place. If nothing work,
# users will have to specify it themselves with PILIQ.set_lib(...)
PILIQ.set_lib(PILIQ._find_library())

# if __name__ == '__main__':
#     img = Image.open('C:/Users/lpio/OneDrive - u-blox/Desktop/rgb.png').convert('RGBA')
#     piq = PILIQ()
#     p, i = piq.quantize(img, 255)
#     Image.fromarray(p[i]).show()
#     p, i = piq.quantize(img, 2)
#     Image.fromarray(p[i]).show()
#     sys.exit(0)
