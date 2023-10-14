# piliq
PIL-LibImageQuant basic and lightweight interface to quantize images. An elementary auto-DLL look-up and loading mechanism is included.

## How it works
A PILIQ instance is loaded by providing a string, which may be a path to an executable, a dll, or a command name. This operation arms the library and enables the caller to quantize RGBA images. The exposed interface is kept to a bare minimum and limited to selected parameters mutually exposed by both pngquant and libimagequant.

### Examples
For pngquant:
```python
piliq = PILIQ("path_to_pngquant.exe")
```

For libimagequant:
```python
piliq = PILIQ("path_to_libimagequant.dll")
```

In either case, this causes the internal wrappers to be initialised once and for all. All subsequent calls to PILIQ constructor will return the library initialised before.

Additionally, PILIQ embeds an elementary autonomous look-up mechanism to find pngquant or libimagequant.
- If you are on Windows x86_64, PILIQ will always manage to at least bind to the provided libimagequant dll.
- Else, PILIQ will attempt to acccess pngquant as a subprocess, assuming it may be in your environment path (quite common for users on Linux or macOS).
- If none of the above stands true, PILIQ looks up libimagequant in known folders collecting dynamic libraries.


To quantize:
```python
img_rgba = Image.open(...).convert('RGB')
#Quantize on a maximum of 223 colors/clusters.
img_p = piliq.quantize(img_rgba, 223)
img_p.save(...)
```

If you wish to work with numpy or bitmaps and palettes, you may set `piliq.return_pil = False`. This will change the return type of quantize() to a tuple containing:
```python
palette, bitmap = img_p
```

### Options
Set the default maximum quality acceptable. Quality is a libimagequant metric bounded between 1 and 100, incl. At 100, libimagequant may always use the highest number of colors, while visually identical results using much less could be obtained in qualities rangin from the 80s to 90s.
```python
piliq.set_quality(quality: int)
```

Set the default speed. Speed is a notion defined by libimagequant and it refers to the algorithms used in the quantization process. The modes ranges from 1 to 10, incl.
```python
piliq.set_speed(speed: int)
```

Set the default maximum number of colors that the result may feature.
The argument of the quantize function overrides the default if provided. Paletted images have a maximum of 256 entries.
```python
piliq.set_default_max_colors(colors: int)
```

Set the dithering level of the output. Libimagequant dithers the quantization results, picking close by colors in the output palette. This does not change the computation speed nor may have any visible impact depending of the settings. The value is a float within [0.0, 1.0], incl.
```python
piliq.set_dithering_level(dithering_level: float)
```


## Limits
- libimagequant is not supported on macOS, only pngquant.
- The internal look-up is rather limited and simple, it may fail everytime. Window python package embeds the x86_64 dll of libimagequant which will auto-load if not provided.
- The provided x86_64 dll is not embedded in frozen "onefile" executables produced via PyInstaller or similar app packagers.
