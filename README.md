# piliq
PIL-LibImageQuant basic quantize interface with simple auto-DLL look-up and loading.

## How it works
First and foremost, you must arm PILIQ with either the pngquant executable or libimagequant.dll / .so files.

For pngquant:
`piliq = PILIQ("path_to_pngquant.exe")`

For libimagequant:
`piliq = PILIQ("path_to_libimagequant.dll")`

In either case, this causes the internal wrappers to be initialised once and for all. All subsequent calls to PILIQ constructor will return the library initialised before.

If you are on Windows x86_64 or have pngquant in path (quite common for users on Linux or macOS), the module also implements basic look-up and will to bind to either pngquant (in PATH) or the libimagequant library (Linux or Windows only). 

Once the library is armed, you can quantize the input Pillow Image with:
`img_out = piliq.quantize(img, colors=127)`

Alternatively, if you wish to work with numpy, bitmaps and palettes, you can set `piliq.return_pil = False`. This will change the return type of quantize to a tuple containing:
`palette, bitmap = img_out`

## Limits
- libimagequant is not supported on macOS (MR are welcome), only pngquant.
- The internal look-up is rather limited and simple, it may fail everytime. Window python package embeds the x86_64 dll of libimagequant which will auto-load if not provided.