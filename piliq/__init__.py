#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MIT License

Copyright (c) 2024 cubicibo

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

class _QuantizrWrapper:
    _lib = None
    def __init__(self) -> None:
        assert self._lib is not None
        self._max_colors = 255
        self._dithering_level = 1.0
        self._attr = self._lib.quantizr_new_options()

        #unused
        self._max_quality = 100
        self._speed = 4

    def set_quality(self, max_quality: int) -> None:
        self._max_quality = max_quality

    def get_quality(self) -> int:
        return 100

    def set_speed(self, speed: int) -> None:
        self._speed = speed

    def get_speed(self) -> int:
        return self._speed

    def set_dithering_level(self, dithering_level: float) -> None:
        self._dithering_level = dithering_level

    def get_dithering_level(self) -> float:
        return self._dithering_level

    def set_default_max_colors(self, colors: int) -> None:
        self._max_colors = colors
        self._lib.quantizr_set_max_colors(self._attr, self._max_colors)

    def get_default_max_colors(self) -> int:
        return self._max_colors

    @classmethod
    def is_ready(cls) -> bool:
        return cls._lib is not None

    @classmethod
    def bind(cls, libp: [ctypes.CDLL | Path | str]) -> bool:
        if isinstance(libp, (Path, str)):
            libp = Path(libp)
            assert libp.exists()
            libp = ctypes.CDLL(libp)
        cls._write_interface(libp)
        cls._lib = libp

    @staticmethod
    def _write_interface(lib: ctypes.CDLL) -> None:
        class ATTR(ctypes.Structure): pass
        class IMAGE(ctypes.Structure): pass
        class RESULT(ctypes.Structure): pass

        class QuantizrPalette(ctypes.Structure):
            _fields_ = [("count", ctypes.c_uint), ("entries", ctypes.c_uint * 256)]

        #attr
        lib.quantizr_new_options.argtype = (None,)
        lib.quantizr_new_options.restype = ctypes.POINTER(ATTR)
        lib.quantizr_free_options.argtype = (ctypes.POINTER(ATTR), )
        lib.quantizr_free_options.restype = None

        #quantize opts
        lib.quantizr_set_max_colors.argtype = (ctypes.POINTER(ATTR), ctypes.c_int)
        lib.quantizr_set_max_colors.restype = ctypes.c_int

        #quantize
        lib.quantizr_create_image_rgba.argtype = (ctypes.c_char_p, ctypes.c_int, ctypes.c_int)
        lib.quantizr_create_image_rgba.restype = ctypes.POINTER(IMAGE)

        lib.quantizr_quantize.argtype = (ctypes.POINTER(IMAGE), ctypes.POINTER(ATTR))
        lib.quantizr_quantize.restype = ctypes.POINTER(RESULT)

        lib.quantizr_get_palette.argtype = (ctypes.POINTER(RESULT),)
        lib.quantizr_get_palette.restype = ctypes.POINTER(QuantizrPalette)

        lib.quantizr_set_dithering_level.argtype = (ctypes.POINTER(RESULT), ctypes.c_float)
        lib.quantizr_set_dithering_level.restype = ctypes.c_int

        lib.quantizr_remap.argtype = (ctypes.POINTER(RESULT), ctypes.POINTER(IMAGE), ctypes.c_char_p, ctypes.c_uint)
        lib.quantizr_remap.restype = ctypes.c_int

        lib.quantizr_free_image.argtype = (ctypes.POINTER(IMAGE),)
        lib.quantizr_free_image.restype = None

        lib.quantizr_free_result.argtype = (ctypes.POINTER(RESULT),)
        lib.quantizr_free_result.restype = None

        _logger.debug("Loaded quantizr.")

    @staticmethod
    def find_library() -> Optional[ctypes.CDLL]:
        LIB_NAME = 'libquantizr'
        if sys.platform.lower() != 'darwin':
            _logger.debug("Not macOS, evading libquantizr loader.")
            return None
        import platform
        carch = platform.machine()
        lib_path = Path(os.path.join(os.path.dirname(sys.modules["piliq"].__file__), f"{LIB_NAME}_{carch}.dylib"))
        if lib_path.exists():
            _logger.debug(f"Found embedded {lib_path} for {carch} in folder, loading library.")
            return ctypes.CDLL(str(lib_path))
        _logger.debug("Did not find embedded quantizr.")
        return None

    def quantize(self, img: Image.Image, colors: Optional[int] = None) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
        if self._attr is None:
            _logger.error("Using a destroyed quantizr instance, aborting.")
            return None, None

        assert img.mode == 'RGBA'
        if colors is not None:
            assert 0 < colors <= 256
            self._lib.quantizr_set_max_colors(self._attr, colors)

        #input_img = np.ascontiguousarray(img, np.uint8).flatten().data.tobytes()
        liq_img = self._lib.quantizr_create_image_rgba(img.tobytes(), int(img.width), int(img.height))
        assert liq_img

        liq_res = self._lib.quantizr_quantize(liq_img, self._attr)
        assert liq_res

        self._lib.quantizr_set_dithering_level(liq_res, ctypes.c_float(self.get_dithering_level()))

        data = bytearray([0]) * (img.width * img.height)
        retval = self._lib.quantizr_remap(liq_res, liq_img, (ctypes.c_char*len(data)).from_buffer(data), len(data))
        if retval == 0:
            data = np.asarray(data, np.uint8).reshape((img.height, img.width))

            palette = self._lib.quantizr_get_palette(liq_res)
            pal = np.zeros((palette.contents.count, 4), np.uint8)
            for entry_id in range(palette.contents.count):
                pal[entry_id, 3] = 0xFF & (palette.contents.entries[entry_id] >> 24)
                pal[entry_id, 2] = 0xFF & (palette.contents.entries[entry_id] >> 16)
                pal[entry_id, 1] = 0xFF & (palette.contents.entries[entry_id] >> 8)
                pal[entry_id, 0] = 0xFF & (palette.contents.entries[entry_id])

        self._lib.quantizr_free_result(liq_res)
        self._lib.quantizr_free_image(liq_img)

        if colors is not None:
            self._lib.quantizr_set_max_colors(self._attr, self._max_colors)

        return (pal, data) if retval == 0 else (None, None)

    def __del__(self) -> None:
        if self._attr is not None:
            self._lib.quantizr_free_options(self._attr)
            _logger.debug("Destroyed quantizr attr handle.")

    def destroy(self) -> None:
        try:
            if self._attr is not None:
                self._lib.quantizr_free_options(self._attr)
                self._attr = None
        except:
            ...

class PNGQuantWrapper:
    _app = 'pngquant'
    _is_win32 = sys.platform == 'win32'
    _is_posix = os.name == 'posix'
    _use_shell = True
    def __init__(self) -> None:
        assert self._app is not None
        assert self.is_ready(), "Cannot call set pngquant executable."
        self._max_quality = 100
        self._speed = 4
        self._dithering_level = 1.0
        self._max_colors = 255

    @classmethod
    def bind(cls, fpath: Union[str, Path]) -> None:
        if  Path(fpath).exists():
            cls._app = fpath
        cls.to_abs_exec()

    @classmethod
    def to_abs_exec(cls) -> None:
        if not cls._use_shell:
            return

        maybe_abs_exec = None
        maybe_path = Path(cls._app)
        if maybe_path.exists():
            maybe_abs_exec = maybe_path.resolve()
        else:
            cmd = None
            if cls._is_win32:
                cmd = ['where.exe']
            elif cls._is_posix:
                cmd = ['command', '-v']
            if cmd is not None:
                prun = subprocess.run(f"{' '.join(cmd)} {cls._app}", capture_output=True, shell=True)
                if cls._is_win32 and prun.returncode:
                    prun = subprocess.run(f"{cmd} {cls._app}.exe", capture_output=True, shell=True)
                path = prun.stdout.split()
                if len(path) > 0 and 0 == prun.returncode:
                    maybe_abs_exec = Path(os.fsdecode(path[0]).strip())
        if isinstance(maybe_abs_exec, Path) and maybe_abs_exec.exists():
            maybe_abs_exec = str(maybe_abs_exec)
            if 0 == cls._test_cmd_version(maybe_abs_exec):
                _logger.debug(f"Found pngquant path '{maybe_abs_exec}', replacing '{cls._app}'")
                cls._app = maybe_abs_exec
                cls._use_shell = False

    @staticmethod
    def _test_cmd_version(exep: str) -> int:
        try:
            proc = subprocess.run([exep, '--version'], capture_output=True, shell=False)
        except FileNotFoundError:
            return -1
        else:
            return proc.returncode

    @classmethod
    def is_ready(cls) -> bool:
        try:
            if cls._use_shell:
                proc = subprocess.run(f"{cls._app} --version", capture_output=True, shell=True)
            else:
                proc = subprocess.run([cls._app, '--version'], capture_output=True, shell=False)
        except FileNotFoundError:
            if cls._is_win32:
                try:
                    if cls._use_shell:
                        proc = subprocess.run(f"{cls._app}.exe --version", capture_output=True, shell=True)
                    else:
                        proc = subprocess.run([cls._app + '.exe', '--version'], capture_output=True, shell=False)
                except FileNotFoundError:
                    ...
                else:
                    if proc.returncode == 0:
                        cls._app += '.exe'
                        return True
        else:
            return proc.returncode == 0
        return False

    def set_quality(self, max_quality: int) -> None:
        self._max_quality = max_quality

    def get_quality(self) -> int:
        return self._max_quality

    def set_speed(self, speed: int) -> None:
        self._speed = speed

    def set_dithering_level(self, dithering_level: float) -> None:
        self._dithering_level = dithering_level

    def get_dithering_level(self) -> float:
        return self._dithering_level

    def set_default_max_colors(self, colors: int) -> None:
        self._max_colors = colors

    def get_default_max_colors(self) -> int:
        return self._max_colors

    def _quantize_posix(self, img: Image.Image, colors: int) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
        w = NamedTemporaryFile()
        with NamedTemporaryFile() as temp_file:
            img.save(temp_file.name, format='png')
            if self._use_shell:
                p = subprocess.Popen(f"{self._app} --quality=0-{self._max_quality} --speed={self._speed} --floyd={self._dithering_level} {colors} -", shell=True, stdout=w, stdin=temp_file, bufsize=0, stderr=subprocess.STDOUT).wait()
            else:
                p = subprocess.Popen([f"{self._app}", f"--quality=0-{self._max_quality}", f"--speed={self._speed}", f"--floyd={self._dithering_level}", f"{colors}", "-"], shell=False, stdout=w, stdin=temp_file, bufsize=0, stderr=subprocess.STDOUT).wait()
        if 0 == p:
            imgp = Image.open(w)
            #PIL palette data is glitched at this point, converting to RGBA fixes it, somehow.
            imgrgba = imgp.convert('RGBA')
            palette = np.reshape(imgp.getpalette('RGBA'), (-1, 4)).astype(np.uint8)
            assert len(palette) <= colors
            bitmap = np.asarray(imgp)
            #Force the existence of RGBA PIL during this chunk of code (?)
            del(imgrgba)
        else:
            _logger.warning(f"pngquant reported error {p}.")
            palette = bitmap = None
        w.close()
        return palette, bitmap

    def quantize(self, img: Image.Image, colors: Optional[int] = None) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
        if colors is None:
            colors = self._max_colors
        else:
            assert 2 <= colors <= 256
        assert img.mode == 'RGBA'

        if __class__._is_posix:
           return self._quantize_posix(img, colors)

        in_tmp, out_tmp = NamedTemporaryFile(delete=False), NamedTemporaryFile(delete=False)
        img.save(in_tmp, format='PNG')

        return_code = -127
        with open(in_tmp.name, 'rb') as inf:
            if self._use_shell:
                p = subprocess.Popen(f"{self._app} --quality=0-{self._max_quality} --speed={self._speed} --floyd={self._dithering_level} {colors} -", shell=True, stdin=inf, stdout=out_tmp, stderr=subprocess.DEVNULL, bufsize=0)
            else:
                p = subprocess.Popen([f"{self._app}", f"--quality=0-{self._max_quality}", f"--speed={self._speed}", f"--floyd={self._dithering_level}", f"{colors}", "-"], shell=False, stdin=inf, stdout=out_tmp, stderr=subprocess.DEVNULL, bufsize=0)
            p.communicate()
            if (return_code := p.returncode) != 0:
                _logger.warning(f"pngquant reported error {return_code}.")
        palette = bitmap = None
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
        try:
            os.unlink(out_tmp.name)
            os.unlink(in_tmp.name)
        except:
            pass
        return palette, bitmap

    def destroy(self):
        pass

class _LIQWrapper:
    _lib = None
    def __init__(self) -> None:
        assert self._lib is not None, "Library must be loaded in the class before instantiating objects."
        self._attr = self._lib.liq_attr_create()
        _logger.debug("Created liq attr handle.")

        self._dithering_level = 1.0
        self._max_colors = 255

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
    def set_dithering_level(self, dithering_level: float) -> None:
        self._dithering_level = dithering_level

    @__ensure_liq
    def get_dithering_level(self) -> float:
        return self._dithering_level

    @__ensure_liq
    def set_speed(self, speed: int) -> None:
        self._lib.liq_set_speed(self._attr, speed)
        assert self._lib.liq_get_speed(self._attr) == speed

    @__ensure_liq
    def get_speed(self) -> int:
        return self._lib.liq_get_speed(self._attr)

    @__ensure_liq
    def set_quality(self, max_quality: int) -> None:
        self._lib.liq_set_quality(self._attr, 0, max_quality)
        assert self._lib.liq_get_max_quality(self._attr) == max_quality

    @__ensure_liq
    def get_quality(self) -> int:
        return self._lib.liq_get_max_quality(self._attr)

    @__ensure_liq
    def set_default_max_colors(self, colors: int) -> None:
        self._max_colors = colors
        self._lib.liq_set_max_colors(self._attr, self._max_colors)

    def get_default_max_colors(self) -> int:
        return self._max_colors

    @__ensure_liq
    def quantize(self, img: Image.Image, colors: Optional[int] = None) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
        if self._attr is None:
            _logger.error("Using a destroyed libimagequant instance, aborting.")
            return None, None

        assert img.mode == 'RGBA'

        if colors is not None:
            assert 0 < colors <= 256
            self._lib.liq_set_max_colors(self._attr, colors)
            assert self._lib.liq_get_max_colors(self._attr) == colors

        liq_img = self._lib.liq_image_create_rgba(self._attr, img.tobytes(), img.width, img.height, 0)

        liq_res = ctypes.c_void_p()
        retval = self._lib.liq_image_quantize(liq_img, self._attr, ctypes.pointer(liq_res))
        if retval != 0:
            _logger.error("libimagequant failed to quantize image.")
        else:
            palette = self._lib.liq_get_palette(liq_res).contents.to_numpy()
            img_quantized = np.zeros((img.height, img.width), np.uint8, 'C')
            self._lib.liq_set_dithering_level(liq_res, ctypes.c_float(self.get_dithering_level()))
            retval = self._lib.liq_write_remapped_image(liq_res, liq_img, img_quantized.ctypes.data_as(ctypes.POINTER(ctypes.c_void_p)), img_quantized.size)
            if retval != 0:
                _logger.error("Failed to retrieve image data from libimagequant.")

        self._lib.liq_result_destroy(liq_res)
        self._lib.liq_image_destroy(liq_img)

        if colors is not None:
            self._lib.liq_set_max_colors(self._attr, self._max_colors)

        return (palette, img_quantized) if retval == 0 else (None, None)
    ####

    @classmethod
    @__ensure_liq
    def get_version(cls) -> int:
        return cls._lib.liq_version()

    def destroy(self) -> None:
        try:
            if self._attr is not None:
                self._lib.liq_attr_destroy(self._attr)
                self._attr = None
        except:
            ...

    @staticmethod
    def find_library() -> Optional[ctypes.CDLL]:
        LIB_NAME = 'libimagequant'
        if os.name == 'nt':
            import platform

            def match_dll_arch(fpath: Path, carch) -> bool:
                # https://stackoverflow.com/a/65586112
                lut_arch = {332: 'I386', 512: 'IA64', 34404: 'x86_64', 452: 'ARM', 43620: 'AARCH64'}
                import struct
                with open(fpath, 'rb') as f:
                    assert f.read(2) == b'MZ'
                    f.seek(60)
                    f.seek(struct.unpack('<L', f.read(4))[0] + 4)
                    return lut_arch.get(struct.unpack('<H', f.read(2))[0], None) == carch
                return False

            carch = platform.machine()
            carch = "x86_64" if carch in ("AMD64", "x86_64") else carch
            lib_path = Path(os.path.join(os.path.dirname(sys.modules["piliq"].__file__), f"{LIB_NAME}_{carch}.dll"))
            if lib_path.exists() and match_dll_arch(lib_path, carch):
                _logger.debug(f"Found embedded {lib_path} for {carch} in folder, loading library.")
                return ctypes.CDLL(str(lib_path))

        elif os.name == 'posix':
            #libimagequant dylib is acting odd on OSX, escape this function.
            if sys.platform.lower() == 'darwin':
                _logger.debug("Evading library look-up, detected macOS.")
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
        class _LIQColor(ctypes.Structure):
            _fields_ = [("r", ctypes.c_char), ("g", ctypes.c_char), ("b", ctypes.c_char), ("a", ctypes.c_char)]

            def to_tuple(self) -> tuple[int, int, int, int]:
                return ord(self.r), ord(self.g), ord(self.b), ord(self.a)

            def to_rgba(self) -> RGBA:
                return RGBA(*self.to_tuple())

        class _LIQPalette(ctypes.Structure):
            _fields_ = [("count", ctypes.c_uint), ("entries", _LIQColor * 256)]

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

        #lib config
        lib.liq_set_speed.argtype = (ctypes.POINTER(liq_attr), ctypes.c_int)
        lib.liq_set_speed.restype = ctypes.c_int
        lib.liq_get_speed.argtype = (ctypes.POINTER(liq_attr))
        lib.liq_get_speed.restype = ctypes.c_int

        lib.liq_set_quality.argtype = (ctypes.POINTER(liq_attr), ctypes.c_int, ctypes.c_int)
        lib.liq_set_quality.restype = ctypes.c_int
        lib.liq_get_max_quality.argtype = (ctypes.POINTER(liq_attr))
        lib.liq_get_max_quality.restype = ctypes.c_int
        lib.liq_get_min_quality.argtype = (ctypes.POINTER(liq_attr))
        lib.liq_get_min_quality.restype = ctypes.c_int

        #quantize
        lib.liq_image_create_rgba.argtype = (ctypes.POINTER(liq_attr), ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_double)
        lib.liq_image_create_rgba.restype = ctypes.POINTER(liq_image)

        lib.liq_image_quantize.argtype = (ctypes.POINTER(liq_image), ctypes.POINTER(liq_attr), ctypes.POINTER(ctypes.POINTER(liq_result)))
        lib.liq_image_quantize.restype = ctypes.c_int

        lib.liq_get_palette.argtype = (ctypes.POINTER(liq_result),)
        lib.liq_get_palette.restype = ctypes.POINTER(_LIQPalette)

        lib.liq_set_dithering_level.argtype = (ctypes.POINTER(liq_result), ctypes.c_float)
        lib.liq_set_dithering_level.restype = ctypes.c_int

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
    def __init__(self, binary: Optional[Union[str, Path]] = None, return_pil: bool = True, /, *, _wrapper = None) -> None:
        self.return_pil = return_pil
        self._wrapped = None

        if _wrapper is not None:
            self._wrapped = _wrapper
            return

        if binary is None:
            _logger.debug("No binary provided, performing look-up.")

            if (is_ready := _LIQWrapper.is_ready()) or (cdl := _LIQWrapper.find_library()) is not None:
                _logger.debug("Detected libimagequant library, using that.")
                if not is_ready:
                    _LIQWrapper.bind(cdl)
                self._wrapped = _LIQWrapper()
            elif PNGQuantWrapper.is_ready():
                PNGQuantWrapper.to_abs_exec()
                _logger.debug("Detected pngquant, using that.")
                self._wrapped = PNGQuantWrapper()
            # quantizr sometime returns garbage, consistently on some image. Do not ever use it by default.
            #elif (is_ready := _QuantizrWrapper.is_ready()) or (cdl := _QuantizrWrapper.find_library()):
            #    _logger.debug("Detected libquantizr, using that.")
            #    if not is_ready:
            #        _QuantizrWrapper.bind(cdl)
            #    self._wrapped = _QuantizrWrapper()
            else:
                raise AssertionError("Could not locate pngquant or libimagequant, aborted.")
        else:
            str_binary =  str(binary).lower()
            if 'pngquant' in str_binary:
                _logger.debug("Executable path seems to be pngquant.")
                PNGQuantWrapper.bind(binary)
                assert PNGQuantWrapper.is_ready()
                self._wrapped = PNGQuantWrapper()
            elif str_binary.split('.')[-1] in ('dll', 'dylib', 'so') and Path(binary).exists():
                _logger.debug("Path seems to be a dynamic library.")
                try:
                    _LIQWrapper.bind(binary)
                except:
                    pass
                else:
                    assert _LIQWrapper.is_ready()
                    self._wrapped = _LIQWrapper()

                if not self._wrapped and sys.platform.lower() == 'darwin':
                    try:
                        _QuantizrWrapper.bind(binary)
                    except:
                        pass
                    else:
                        assert _QuantizrWrapper.is_ready()
                        self._wrapped = _QuantizrWrapper()

                if not self._wrapped:
                    raise AssertionError(f"Cannot identify the dynamic library binary '{binary}'.")
            else:
                raise AssertionError("Cannot identify the loaded binary file.")

    def quantize(self, img: Image.Image, colors: Optional[int] = None) -> Union[Image.Image, tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]]:
        assert self._wrapped is not None, "Destroyed instance."
        pal, oimg = self._wrapped.quantize(img, colors)
        if self.return_pil:
            return Image.fromarray(pal[oimg], 'RGBA')
        return pal, oimg

    def is_ready(self) -> bool:
        if self._wrapped is not None:
            return self._wrapped.is_ready()
        return False

    def set_quality(self, max_quality: int) -> None:
        assert 0 < max_quality <= 100
        if self._wrapped is not None:
            self._wrapped.set_quality(max_quality)

    def set_speed(self, speed: int) -> None:
        assert 1 <= speed <= 10
        if self._wrapped is not None:
            self._wrapped.set_speed(speed)

    def set_dithering_level(self, dithering_level: float) -> None:
        assert 0 <= dithering_level <= 1.0
        if self._wrapped is not None:
            self._wrapped.set_dithering_level(float(dithering_level))

    def set_default_max_colors(self, colors: 255) -> None:
        assert 2 <= colors <= 256
        if self._wrapped is not None:
            self._wrapped.set_default_max_colors(colors)

    def get_quality(self) -> Optional[int]:
        if self._wrapped is not None:
            return self._wrapped.get_quality()

    def get_dithering_level(self) -> Optional[float]:
        if self._wrapped is not None:
            return self._wrapped.get_dithering_level()

    def get_speed(self) -> Optional[int]:
        if self._wrapped is not None:
            return self._wrapped.get_speed()

    @property
    def lib_name(self) -> str:
        if isinstance(self._wrapped, PNGQuantWrapper):
            return 'pngquant'
        elif isinstance(self._wrapped, _LIQWrapper):
            return 'libimagequant'
        elif isinstance(self._wrapped, _QuantizrWrapper):
            return 'quantizr'
        else:
            _logger.debug("PILIQ is not armed with any quantization library.")
            return None

    def __del__(self):
        self.destroy()

    def destroy(self) -> None:
        if self._wrapped is not None:
            self._wrapped.destroy()
            self._wrapped = None

    @staticmethod
    def set_log_level(level: int) -> None:
        prev_level = _logger.level
        _logger.setLevel(level)
        _logger.debug(f"Changed logging level from {prev_level} to {level}.")
####
