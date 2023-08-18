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
import subprocess

from typing import Union, Optional
from functools import wraps

from dataclasses import dataclass
from tempfile import NamedTemporaryFile
from pathlib import Path

from PIL import Image
from numpy import typing as npt
import numpy as np

def get_logger(name: str = 'piliq',
               level: int = logging.INFO,
               format_str: str = ' %(name)s: %(levelname).4s : %(message)s') -> logging.Logger:
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
####

class _PNGQuantWrapper:
    _app = 'pngquant'
    _is_win32 = sys.platform == 'win32'
    def __init__(self) -> None:
        assert self._app is not None
        assert self.is_ready(), "Cannot call set pngquant executable."

    @classmethod
    def bind(cls, fpath: Union[str, Path]) -> None:
        if  Path(fpath).exists():
            cls._app = fpath

    @classmethod
    def is_ready(cls) -> bool:
        try:
            proc = subprocess.run([cls._app, '--version'], capture_output=True)
        except FileNotFoundError:
            if cls._is_win32:
                try:
                    proc = subprocess.run([cls._app + '.exe', '--version'], capture_output=True)
                except FileNotFoundError:
                    ...
                else:
                    if proc.returncode == 0:
                        cls._app += '.exe'
                        return True
        else:
            return proc.returncode == 0
        return False

    def _quantize_posix(self, img: Image.Image, colors: int = 255) -> ...:
        w = NamedTemporaryFile()
        with NamedTemporaryFile() as temp_file:
            img.save(temp_file.name, format='png')
            p = subprocess.Popen(f"{self._app} {colors} -", shell=True, stdout=w, stdin=temp_file, bufsize=0, stderr=subprocess.STDOUT).wait()
        imgp = Image.open(w)
        #PIL palette data is glitched at this point, converting to RGBA fixes it, somehow.
        imgrgba = imgp.convert('RGBA')
        palette = np.reshape(imgp.getpalette('RGBA'), (-1, 4)).astype(np.uint8)
        assert len(palette) <= colors
        bitmap = np.asarray(imgp)
        #Force the existence of RGBA PIL during this chunk of code (?)
        del(imgrgba)
        w.close()
        return palette, bitmap

    def quantize(self, img: Image.Image, colors: int = 255) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
        if os.name == 'posix':
           return self._quantize_posix(img, colors)

        assert img.mode == 'RGBA'
        assert 0 < colors <= 256

        in_tmp, out_tmp = NamedTemporaryFile(), NamedTemporaryFile()
        if self._is_win32:
            in_tmp.close()
        img.save(in_tmp, format='PNG')

        return_code = -127
        with open(in_tmp.name, 'rb') as inf:
            p = subprocess.Popen(f"{self._app} {colors} -", shell=True, stdin=inf, stdout=out_tmp, stderr=subprocess.DEVNULL, bufsize=0)
            p.communicate()
            if (return_code := p.returncode) != 0:
                _logger.warning(f"pngquant reported error {return_code}.")
        palette, bitmap = None, None
        if return_code == 0:
            imgp = Image.open(out_tmp)
            imgrgba = imgp.convert('RGBA')

            palette = np.reshape(imgp.getpalette("RGBA"), (-1, 4)).astype(np.uint8)
            assert len(palette) <= colors
            bitmap = np.asarray(imgp)
            imgp.close()
            del(imgrgba)
        out_tmp.close()
        in_tmp.close()
        if self._is_win32:
            os.unlink(out_tmp.name)
            os.unlink(in_tmp.name)
        return palette, bitmap

    def destroy(self):
        pass

class _LIQWrapper:
    _lib = None
    def __init__(self) -> None:
        assert self._lib is not None, "Library must be loaded in the class before instantiating objects."
        self._attr = self._lib.liq_attr_create()
        _logger.debug("Created liq attr handle.")

    def __del__(self) -> None:
        if self._attr is not None:
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
    def bind(cls, lib: Union[str, Path, ctypes.CDLL]) -> None:
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
        if self._attr is None:
            _logger.error("Using a destroyed libimagequant instance, aborting.")
            return None, None

        assert img.mode == 'RGBA'
        assert 0 < colors <= 256
        self._lib.liq_set_max_colors(self._attr, colors)
        assert self._lib.liq_get_max_colors(self._attr) == colors

        img_array = np.ascontiguousarray(img, np.uint8)
        liq_img = self._lib.liq_image_create_rgba(self._attr, img_array.ctypes.data_as(ctypes.POINTER(ctypes.c_void_p)), img.width, img.height, 0)

        liq_res = ctypes.c_void_p()
        retval = self._lib.liq_image_quantize(liq_img, self._attr, ctypes.pointer(liq_res))
        if retval != 0:
            _logger.error("libimagequant failed to quantize image.")
        else:
            palette = self._lib.liq_get_palette(liq_res).contents.to_numpy()
            img_quantized = np.zeros((img.height, img.width), np.uint8, 'C')

            retval = self._lib.liq_write_remapped_image(liq_res, liq_img, img_quantized.ctypes.data_as(ctypes.POINTER(ctypes.c_void_p)), img_quantized.size)
            if retval != 0:
                _logger.error("Failed to retrieve image data from libimagequant.")

        self._lib.liq_result_destroy(liq_res)
        self._lib.liq_image_destroy(liq_img)
        return (palette, img_quantized) if retval == 0 else (None, None)
    ####

    @classmethod
    @__ensure_liq
    def get_version(cls) -> int:
        return cls._lib.liq_version()

    @__ensure_liq
    def destroy(self) -> None:
        if self._attr is not None:
            self._lib.liq_attr_destroy(self._attr)
            self._attr = None

    @staticmethod
    def find_library() -> Optional[ctypes.CDLL]:
        LIB_NAME = 'libimagequant'
        if os.name == 'nt':
            import platform

            def match_dll_arch(fpath: Path) -> bool:
                # https://stackoverflow.com/a/65586112
                lut_arch = {332: 'I386', 512: 'IA64', 34404: 'AMD64', 452: 'ARM', 43620: 'AARCH64'}
                import struct
                with open(fpath, 'rb') as f:
                    assert f.read(2) == b'MZ'
                    f.seek(60)
                    f.seek(struct.unpack('<L', f.read(4))[0] + 4)
                    return lut_arch.get(struct.unpack('<H', f.read(2))[0], None) == platform.machine()
                return False

            lib_path = Path(os.path.join(os.path.dirname(sys.modules["piliq"].__file__), f"{LIB_NAME}_{platform.machine()}.dll"))
            if lib_path.exists() and match_dll_arch(lib_path):
                _logger.debug(f"Found embedded {lib_path} in folder, loading library.")
                return ctypes.CDLL(str(lib_path))

        elif os.name == 'posix':
            #libimagequant dylib is acting odd on OSX, escape this function.
            if sys.platform.lower() == 'darwin':
                _logger.debug(f"Evading library look-up, detected macOS.")
                return None

            retry_linux = False
            try:
                ret = subprocess.run(['brew', '--prefix', LIB_NAME], capture_output=True)
            except FileNotFoundError:
                has_brew = False
            else:
                has_brew = ret.stderr == b'' and ret.stdout != b''
            if not has_brew and sys.platform.lower() != 'darwin':
                retry_linux = True
            elif has_brew and ret.stdout != b'':
                if (lib_file := Path(str(ret.stdout, 'utf-8').strip()).joinpath('lib', LIB_NAME+'.dylib')).exists():
                    _logger.debug(f"Found {lib_file}, loading library.")
                    return ctypes.CDLL(lib_file)

            if retry_linux:
                ret = subprocess.run(['dpkg', '-L', LIB_NAME+'-dev'], capture_output=True)
                for fpath in map(lambda bpath: Path(str(bpath, 'utf-8')), ret.stdout.split(b'\n')):
                    if str(fpath).endswith('.so') and fpath.exists():
                        _logger.debug(f"Found {fpath}, loading as shared object.")
                        return ctypes.CDLL(fpath)
        else:
            _logger.debug(f"Unknown OS, {LIB_NAME} has to be loaded manually.")
            return None
        _logger.debug(f"Failed to bind to {LIB_NAME}. Library must be set manually.")
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

class PILIQ:
    def __init__(self, binary: Optional[Union[str, Path]] = None, return_pil: bool = True) -> None:
        if binary is None:
            _logger.debug("No binary provided, performing look-up.")
            if _PNGQuantWrapper.is_ready():
                _logger.debug("Detected pngquant, using that.")
                self._wrapped = _PNGQuantWrapper()
            elif (is_ready := _LIQWrapper.is_ready()) or (cdl := _LIQWrapper.find_library()) is not None:
                _logger.debug("Detected libimagequant library, using that.")
                if not is_ready:
                    _LIQWrapper.bind(cdl)
                self._wrapped = _LIQWrapper()
            else:
                raise AssertionError("Could not locate pngquant or libimagequant, aborted.")
        else:
            str_binary =  str(binary).lower()
            if 'pngquant' in str_binary:
                _logger.debug("Executable path seems to be pngquant.")
                _PNGQuantWrapper.bind(binary)
                assert _PNGQuantWrapper.is_ready()
                self._wrapped = _PNGQuantWrapper()
            elif str_binary.split('.')[-1] in ('dll', 'dylib', 'so'):
                _logger.debug("Path seems to be libimagequant dynamic library.")
                _LIQWrapper.bind(binary)
                assert _LIQWrapper.is_ready()
                self._wrapped = _LIQWrapper()
            else:
                raise AssertionError("Cannot identify the loaded binary file.")
        self.return_pil = return_pil

    def quantize(self, img: Image.Image, colors: int = 255) -> Union[Image.Image, tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]]:
        assert self._wrapped is not None, "Destroyed instance."
        pal, oimg = self._wrapped.quantize(img, colors)
        if self.return_pil:
            return Image.fromarray(pal[oimg], 'RGBA')
        return pal, oimg

    @property
    def lib_name(self) -> str:
        if isinstance(self._wrapped, _PNGQuantWrapper):
            return 'pngquant'
        elif isinstance(self._wrapped, _LIQWrapper):
            return 'libimagequant'
        else:
            _logger.debug("PILIQ is not armed with any quantization library.")
            return None

    def destroy(self) -> None:
        if self._wrapped is not None:
            self._wrapped.destroy()
            self._wrapped = None

    def set_log_level(self, level: int) -> None:
        prev_level = _logger.level
        _logger.setLevel(level)
        _logger.debug(f"Changed logging level from {prev_level} to {level}.")
####
