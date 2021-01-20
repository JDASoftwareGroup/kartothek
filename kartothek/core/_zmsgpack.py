# -*- coding: utf-8 -*-
import msgpack

try:
    import zstandard as zstd
except ImportError:
    # zstandard < 0.15.0
    import zstd  # type: ignore


def packb(obj):
    cctx = zstd.ZstdCompressor(write_content_size=True)
    compressed = cctx.compress(msgpack.packb(obj))
    return compressed


def unpackb(bts):
    dctx = zstd.ZstdDecompressor()
    decompressed = dctx.decompress(bts)
    return msgpack.unpackb(decompressed, raw=False)
