"""
[`GridIntCodec`][numcodecs_grid_int.GridIntCodec] for the [`numcodecs`][numcodecs] buffer compression API.
"""

__all__ = ["GridIntCodec"]

import math
from collections.abc import Callable
from functools import reduce
from io import BytesIO

import leb128
import numcodecs.compat
import numcodecs.registry
import numpy as np
from numcodecs.abc import Codec
from numcodecs_combinators.abc import CodecCombinatorMixin
from typing_extensions import Buffer  # MSPV 3.12


class GridIntCodec(Codec, CodecCombinatorMixin):
    """
    Meta-codec that losslessly maps floating-point data that lies on a uniform
    grid `offset + k * scale` to the integer indices `k`, which are encoded with
    the `codec`.

    Many datasets are stored with a fixed quantisation, e.g. GRIB simple
    packing produces values `reference + k * 2**binary_scale` and data derived
    from single-precision inputs lies on a `2**-23` grid. Encoding such data as
    small integers is lossless and typically compresses much better than the
    floating-point representation.

    If `offset` and `scale` are not given, they are detected from the data:
    `offset` is the minimum finite value and `scale` is the smallest positive
    difference between two distinct finite values. Encoding verifies that
    `offset + k * scale` reproduces every finite value *bitwise*, and raises a
    [`ValueError`][ValueError] otherwise.

    Non-finite values (NaN, +-inf) are mapped to the index `0` and thus decode
    to the grid `offset`. To preserve them, combine this codec with a masking
    meta-codec such as
    [`numcodecs_mask.MaskMetaCodec`](https://numcodecs-mask.readthedocs.io).

    Parameters
    ----------
    codec : dict | Codec
        The configuration or instantiated codec that encodes the integer
        indices.
    offset : None | float, optional
        The grid offset, or [`None`][None] to detect it from the data.
    scale : None | float, optional
        The positive grid spacing, or [`None`][None] to detect it from the
        data.
    """

    __slots__: tuple[str, ...] = ("_codec", "_offset", "_scale")
    _codec: Codec
    _offset: None | float
    _scale: None | float

    codec_id: str = "grid_int"  # type: ignore

    def __init__(
        self,
        *,
        codec: dict | Codec,
        offset: None | float = None,
        scale: None | float = None,
    ) -> None:
        if offset is not None and not math.isfinite(offset):
            raise ValueError("offset must be finite")
        if scale is not None and not (math.isfinite(scale) and scale > 0):
            raise ValueError("scale must be finite and positive")

        self._codec = (
            codec if isinstance(codec, Codec) else numcodecs.registry.get_codec(codec)
        )
        self._offset = None if offset is None else float(offset)
        self._scale = None if scale is None else float(scale)

    @staticmethod
    def detect(a: np.ndarray) -> tuple[float, float]:
        """
        Detect the grid `offset` and `scale` of the finite values in `a`.

        Parameters
        ----------
        a : np.ndarray
            The floating-point data array.

        Returns
        -------
        offset : float
            The minimum finite value (or `0.0` if there are no finite values).
        scale : float
            The smallest positive difference between distinct finite values
            (or `1.0` if there are fewer than two distinct finite values).
        """

        unique = np.unique(a[np.isfinite(a)].astype(np.float64))

        if unique.size == 0:
            return 0.0, 1.0
        if unique.size == 1:
            return float(unique[0]), 1.0

        return float(unique[0]), float(np.diff(unique).min())

    def encode(self, buf: Buffer) -> bytes:
        """
        Encode the data in `buf`.

        Parameters
        ----------
        buf : Buffer
            Floating-point data to be encoded. May be any object supporting
            the new-style buffer protocol.

        Returns
        -------
        enc : bytes
            Encoded data as a bytestring.
        """

        a = numcodecs.compat.ensure_ndarray(buf)
        dtype, shape = a.dtype, a.shape

        if not np.issubdtype(dtype, np.floating):
            raise TypeError("can only encode floating point values")

        is_finite = np.isfinite(a)

        offset, scale = self._offset, self._scale
        if offset is None or scale is None:
            detected_offset, detected_scale = self.detect(a)
            offset = detected_offset if offset is None else offset
            scale = detected_scale if scale is None else scale

        indices = np.where(
            is_finite, np.rint((a.astype(np.float64) - offset) / scale), 0.0
        )

        reconstructed = (offset + indices * scale).astype(dtype)
        bits = dtype.str.replace("f", "u")
        if not np.array_equal(
            reconstructed[is_finite].view(bits), a[is_finite].view(bits)
        ):
            raise ValueError(
                f"data is not exactly representable on the grid {offset} + k * {scale}"
            )

        index_min = int(indices.min()) if indices.size > 0 else 0
        index_max = int(indices.max()) if indices.size > 0 else 0
        index_dtype: np.dtype
        for candidate in (np.uint8, np.uint16, np.uint32, np.int64):
            info = np.iinfo(candidate)
            if index_min >= info.min and index_max <= info.max:
                index_dtype = np.dtype(candidate)
                break
        else:  # pragma: no cover
            raise ValueError("grid indices exceed the int64 range")

        encoded = numcodecs.compat.ensure_ndarray(
            self._codec.encode(indices.astype(index_dtype))
        )

        # message: dtype shape offset scale index-dtype
        #          encoded-dtype encoded-shape [padding] encoded
        message: list[bytes | bytearray] = []

        message.append(leb128.u.encode(len(dtype.str)))
        message.append(dtype.str.encode("ascii"))

        message.append(leb128.u.encode(len(shape)))
        for s in shape:
            message.append(leb128.u.encode(s))

        message.append(np.array([offset, scale], dtype="<f8").tobytes())

        message.append(leb128.u.encode(len(index_dtype.str)))
        message.append(index_dtype.str.encode("ascii"))

        message.append(leb128.u.encode(len(encoded.dtype.str)))
        message.append(encoded.dtype.str.encode("ascii"))

        message.append(leb128.u.encode(encoded.ndim))
        for s in encoded.shape:
            message.append(leb128.u.encode(s))

        # insert padding to align with encoded itemsize
        message.append(
            b"\0"
            * (
                encoded.dtype.itemsize
                - (sum(len(m) for m in message) % encoded.itemsize)
            )
        )

        # ensure that the encoded values are encoded in little endian binary
        message.append(encoded.astype(encoded.dtype.newbyteorder("<")).tobytes())

        return b"".join(message)

    def decode(self, buf: Buffer, out: None | Buffer = None) -> Buffer:
        """
        Decode the data in `buf`.

        Parameters
        ----------
        buf : Buffer
            Encoded data. Must be an object representing a bytestring, e.g.
            [`bytes`][bytes] or a 1D array of [`np.uint8`][numpy.uint8]s etc.
        out : Buffer, optional
            Writeable buffer to store decoded data. N.B. if provided, this
            buffer must be exactly the right size to store the decoded data.

        Returns
        -------
        dec : Buffer
            Decoded data. May be any object supporting the new-style buffer
            protocol.
        """

        b = numcodecs.compat.ensure_bytes(buf)

        b_io = BytesIO(b)

        # message: dtype shape offset scale index-dtype
        #          encoded-dtype encoded-shape [padding] encoded
        dtype = np.dtype(b_io.read(leb128.u.decode_reader(b_io)[0]).decode("ascii"))
        shape = tuple(
            leb128.u.decode_reader(b_io)[0]
            for _ in range(leb128.u.decode_reader(b_io)[0])
        )

        offset, scale = np.frombuffer(b_io.read(16), dtype="<f8", count=2)

        index_dtype = np.dtype(
            b_io.read(leb128.u.decode_reader(b_io)[0]).decode("ascii")
        )

        encoded_dtype = np.dtype(
            b_io.read(leb128.u.decode_reader(b_io)[0]).decode("ascii")
        )
        encoded_shape = tuple(
            leb128.u.decode_reader(b_io)[0]
            for _ in range(leb128.u.decode_reader(b_io)[0])
        )
        encoded_size = reduce(lambda a, b: a * b, encoded_shape, 1)

        # remove padding to align with encoded itemsize
        b_io.read(encoded_dtype.itemsize - (b_io.tell() % encoded_dtype.itemsize))

        encoded = (
            np.frombuffer(
                b_io.read(encoded_size * encoded_dtype.itemsize),
                dtype=encoded_dtype.newbyteorder("<"),
                count=encoded_size,
            )
            .astype(encoded_dtype)
            .reshape(encoded_shape)
        )

        indices = np.empty(shape, dtype=index_dtype)
        indices_decoded = self._codec.decode(encoded, out=indices)
        indices = numcodecs.compat.ensure_ndarray(indices_decoded).reshape(shape)

        decoded = (float(offset) + indices.astype(np.float64) * float(scale)).astype(
            dtype
        )

        return numcodecs.compat.ndarray_copy(decoded, out)  # type: ignore

    def get_config(self) -> dict:
        """
        Returns the configuration of this grid-int meta-codec.

        [`numcodecs.registry.get_codec(config)`][numcodecs.registry.get_codec]
        can be used to reconstruct this codec from the returned config.

        Returns
        -------
        config : dict
            Configuration of this grid-int meta-codec.
        """

        return dict(
            id=type(self).codec_id,
            codec=self._codec.get_config(),
            offset=self._offset,
            scale=self._scale,
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(codec={self._codec!r}, offset={self._offset!r}, scale={self._scale!r})"

    def map(self, mapper: Callable[[Codec], Codec]) -> "GridIntCodec":
        """
        Apply the `mapper` to the inner `codec` of this grid-int meta-codec.

        Parameters
        ----------
        mapper : Callable[[Codec], Codec]
            The callable that is applied to the inner codec.

        Returns
        -------
        mapped : GridIntCodec
            The mapped grid-int meta-codec.
        """

        return GridIntCodec(
            codec=mapper(self._codec), offset=self._offset, scale=self._scale
        )


numcodecs.registry.register_codec(GridIntCodec)
