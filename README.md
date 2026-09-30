[![image](https://img.shields.io/github/actions/workflow/status/SF-N/numcodecs-grid-int/ci.yml?branch=main)](https://github.com/SF-N/numcodecs-grid-int/actions/workflows/ci.yml?query=branch%3Amain)
[![image](https://img.shields.io/pypi/v/numcodecs-grid-int.svg)](https://pypi.python.org/pypi/numcodecs-grid-int)
[![image](https://img.shields.io/pypi/l/numcodecs-grid-int.svg)](https://github.com/SF-N/numcodecs-grid-int/blob/main/LICENSE)
[![image](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2FSF-N%2Fnumcodecs-grid-int%2Frefs%2Fheads%2Fmain%2Fpyproject.toml)](https://pypi.python.org/pypi/numcodecs-grid-int)
[![image](https://readthedocs.org/projects/numcodecs-grid-int/badge/?version=latest)](https://numcodecs-grid-int.readthedocs.io/en/latest/?badge=latest)

# numcodecs-grid-int

`GridIntCodec` for the [`numcodecs`] buffer compression API.

The `GridIntCodec` is a meta-codec for floating-point data that lies on a uniform grid `offset + k * scale`, e.g. GRIB simple packing (`reference + k * 2**binary_scale`) or data derived from single-precision inputs (`2**-23` grid). The data is losslessly mapped to the integer indices `k`, which are encoded with an inner codec and typically compress much better than the floating-point representation. The grid is detected from the data unless `offset` and `scale` are given; encoding verifies that every finite value is reproduced *bitwise* and raises a `ValueError` otherwise.

```python
from numcodecs_grid_int import GridIntCodec

codec = GridIntCodec(codec=dict(id="zstd", level=19))
```

Non-finite values decode to the grid `offset`; combine with a masking meta-codec such as [`numcodecs-mask`](https://numcodecs-mask.readthedocs.io) to preserve them.

[`numcodecs`]: https://numcodecs.readthedocs.io/en/stable/

## License

Licensed under the Mozilla Public License, Version 2.0 ([LICENSE](LICENSE) or https://www.mozilla.org/en-US/MPL/2.0/).


## Funding

The `numcodecs-grid-int` package has been developed as part of [ESiWACE3](https://www.esiwace.eu), the third phase of the Centre of Excellence in Simulation of Weather and Climate in Europe.

Funded by the European Union. This work has received funding from the European High Performance Computing Joint Undertaking (JU) under grant agreement No 101093054.
