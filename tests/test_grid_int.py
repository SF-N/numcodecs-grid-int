import numcodecs
import numcodecs.registry
import numpy as np
import pytest

INNER = dict(id="zlib", level=1)


def test_from_config():
    codec = numcodecs.registry.get_codec(dict(id="grid_int", codec=INNER))
    assert codec.__class__.__name__ == "GridIntCodec"
    assert codec.__class__.__module__ == "numcodecs_grid_int"
    assert codec.get_config() == dict(
        id="grid_int", codec=INNER, offset=None, scale=None
    )


def test_invalid():
    from numcodecs_grid_int import GridIntCodec

    with pytest.raises(ValueError):
        GridIntCodec(codec=INNER, scale=0.0)
    with pytest.raises(ValueError):
        GridIntCodec(codec=INNER, offset=np.inf)
    with pytest.raises(TypeError):
        GridIntCodec(codec=INNER).encode(np.arange(10))
    # data that does not lie on a uniform grid
    with pytest.raises(ValueError):
        GridIntCodec(codec=INNER).encode(np.random.default_rng(0).random(1000))
    # data that lies on a grid, but not on the given one
    with pytest.raises(ValueError):
        GridIntCodec(codec=INNER, offset=0.0, scale=0.3).encode(np.arange(10.0))


def test_detect():
    from numcodecs_grid_int import GridIntCodec

    assert GridIntCodec.detect(np.array([1.0, 3.0, 5.0, np.nan])) == (1.0, 2.0)
    assert GridIntCodec.detect(np.array([2.5])) == (2.5, 1.0)
    assert GridIntCodec.detect(np.array([np.nan])) == (0.0, 1.0)


def test_map():
    from numcodecs_grid_int import GridIntCodec

    codec = GridIntCodec(codec=INNER, offset=1.0)
    mapped = codec.map(lambda c: numcodecs.registry.get_codec(dict(id="zlib", level=9)))
    assert mapped.get_config() == dict(
        id="grid_int", codec=dict(id="zlib", level=9), offset=1.0, scale=None
    )


def check_roundtrip(data: np.ndarray, **kwargs):
    codec = numcodecs.registry.get_codec(dict(id="grid_int", codec=INNER, **kwargs))

    encoded = codec.encode(data)
    decoded = np.asarray(codec.decode(encoded))

    assert decoded.dtype == data.dtype
    assert decoded.shape == data.shape

    finite = np.isfinite(data)
    bits = data.dtype.str.replace("f", "u")
    np.testing.assert_array_equal(decoded[finite].view(bits), data[finite].view(bits))

    out = np.empty_like(data)
    codec.decode(encoded, out=out)
    np.testing.assert_array_equal(out, decoded)

    return encoded


def test_roundtrip():
    rng = np.random.default_rng(42)

    # integer-valued floating point data
    check_roundtrip(rng.integers(0, 20, size=(50, 60)).astype(np.float64))
    check_roundtrip(rng.integers(-5, 5, size=(50, 60)).astype(np.float32))

    # single-precision fractions in [0, 1] on a 2**-23 grid, stored as float64
    data = (rng.integers(0, 2**23 + 1, size=10000) * 2.0**-23).astype(np.float32)
    check_roundtrip(data.astype(np.float64))
    check_roundtrip(data)

    # GRIB-like packing: reference + k * 2**binary_scale
    data = 273.15 + rng.integers(0, 2**16, size=(100, 100)) * 2.0**-7
    check_roundtrip(data)
    check_roundtrip(data, offset=273.15, scale=2.0**-7)

    # non-finite values decode to the offset
    data = np.array([1.0, 2.0, np.nan, 3.0, np.inf])
    codec = numcodecs.registry.get_codec(dict(id="grid_int", codec=INNER))
    decoded = np.asarray(codec.decode(codec.encode(data)))
    np.testing.assert_array_equal(decoded, [1.0, 2.0, 1.0, 3.0, 1.0])

    # degenerate cases
    check_roundtrip(np.full((3, 3), 7.5))
    check_roundtrip(np.zeros((0,), dtype=np.float64))
    check_roundtrip(np.array(1.0))


def test_small_index_dtype():
    # up to 256 distinct values fit into uint8 indices, which compress well
    data = np.arange(256, dtype=np.float64) * 0.25
    encoded = check_roundtrip(data)
    assert len(encoded) < data.nbytes / 4
