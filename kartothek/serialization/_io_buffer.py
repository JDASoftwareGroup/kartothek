"""
Improved IO buffering compared to ``io.BufferedReader``.

The main issues w/ ``io.BufferedReader`` is that it is only meant of sequencial reads and that it resets the buffer on
``.seek(...)``. This happens quite often in pyarrow and basically renders the buffereing inefficient.
"""
import io
import logging

_logger = logging.getLogger(__name__)


class BlockBuffer(io.BufferedIOBase):
    """
    Block-based buffer.

    The input is split into fixed sizes blocks. Every block can be read independently.
    """

    def __init__(self, raw, blocksize=1024):
        self._raw = raw
        self._blocksize = blocksize
        self._size = None
        self._cached_blocks = None
        self._pos = 0

        if self._raw_closed():
            raise ValueError("Cannot use closed file object")
        if not self._raw_readable():
            raise ValueError("raw must be readable")
        if not self._raw_seekable():
            raise ValueError("raw must be seekable")
        if blocksize < 1:
            raise ValueError("blocksize must be at least 1")

    def _raw_closed(self):
        """
        If supported by ``raw``, return its closed state, otherwise return ``False``.
        """
        if hasattr(self._raw, "closed"):
            return self._raw.closed
        else:
            return False

    def _raw_readable(self):
        """
        If supported by ``raw``, return its readable state, otherwise return ``True``.
        """
        if hasattr(self._raw, "readable"):
            return self._raw.readable()
        else:
            return True

    def _raw_seekable(self):
        """
        If supported by ``raw``, return its seekable state, otherwise return ``True``.
        """
        if hasattr(self._raw, "seekable"):
            return self._raw.seekable()
        else:
            return True

    def _setup_cache(self):
        """
        Set up cache data structure and inspect underlying IO object.

        If the cache is already inialized, this is a no-op.
        """
        if self._cached_blocks is not None:
            # cache initialized, nothing to do
            return

        if hasattr(self._raw, "size"):
            self._size = self._raw.size
        elif hasattr(self._raw, "__len__"):
            self._size = len(self._raw)
        else:
            self._raw.seek(0, 2)
            self._size = self._raw.tell()

        n_blocks = self._size // self._blocksize
        if self._size % self._blocksize:
            n_blocks += 1

        self._cached_blocks = [None] * n_blocks

    def _fetch_blocks(self, block, n):
        """
        Fetch blocks from underlying IO object.

        This will mark the fetched blocks as loaded.

        Parameters
        ----------
        block: int
            First block to fetch.
        n: int
            Number of blocks to fetch.
        """
        assert n > 0

        # seek source
        offset = self._blocksize * block
        self._raw.seek(offset, 0)

        # read data into temporary variable and dump it into cache
        size = min(self._blocksize * n, self._size - offset)
        data = self._raw.read(size)
        if len(data) != size:
            err = (
                f"Expected raw read to return {size} bytes, but instead got {len(data)}"
            )
            _logger.error(err)
            raise AssertionError(err)

        # fill blocks
        for i in range(n):
            begin = i * self._blocksize
            end = min((i + 1) * self._blocksize, size)
            self._cached_blocks[block + i] = data[begin:end]

    def _ensure_range_loaded(self, start, size):
        """
        Ensure that a given byte range is loaded into the cache.

        This will scan for blocks that are not loaded yet and tries to load consecutive blocks as once.

        Parameters
        ----------
        start: int
            First byte of the range.
        size: int
            Number of bytes in the range.
        """
        if size < 0:
            msg = f"Expected size >= 0, but got start={start}, size={size}"
            _logger.error(msg)
            raise AssertionError(msg)

        block = start // self._blocksize
        offset = start % self._blocksize

        # iterate over blocks in range and figure out long sub-ranges of blocks to fetch at once
        done = -offset
        to_fetch_start = None
        to_fetch_n = None
        while done < size:
            if self._cached_blocks[block] is not None:
                # current block is loaded
                if to_fetch_start is not None:
                    # there was a block range to be loaded, do that now
                    self._fetch_blocks(to_fetch_start, to_fetch_n)

                    # no active block range anymore
                    to_fetch_start = None
                    to_fetch_n = None
            else:
                # current block is missing, do we already have a block range to append to?
                if to_fetch_start is None:
                    # no block range open, create a new one
                    to_fetch_start = block
                    to_fetch_n = 1
                else:
                    # current block range exists, append block
                    to_fetch_n += 1

            done += self._blocksize
            block += 1

        if to_fetch_start is not None:
            # this is the last active block range, fetch it
            self._fetch_blocks(to_fetch_start, to_fetch_n)

    def _read_data_from_blocks(self, start, size):
        """
        Read data from bytes.

        Parameters
        ----------
        start: int
            First byte of the range.
        size: int
            Number of bytes in the range.

        Returns
        -------
        data: bytes
            Requested data
        """
        block = start // self._blocksize
        offset = start % self._blocksize

        read_size = size + offset
        n_blocks = read_size // self._blocksize
        if read_size % self._blocksize != 0:
            n_blocks += 1

        return b"".join(self._cached_blocks[block : (block + n_blocks)])[
            offset : (size + offset)
        ]

    def _check_closed(self):
        """
        Check that file object currently is not closed.

        Raises
        ------
        ValueError: in case file object is closed
        """
        if self.closed:
            raise ValueError("I/O operation on closed file.")

    def read(self, size=None):
        self._check_closed()
        self._setup_cache()

        if (size is None) or (size < 0) or (self._pos + size > self._size):
            # read entire, remaining file
            size = self._size - self._pos

        self._ensure_range_loaded(self._pos, size)
        result = self._read_data_from_blocks(self._pos, size)
        self._pos += size
        return result

    def tell(self):
        self._check_closed()
        return self._pos

    def seek(self, offset, whence=0):
        self._check_closed()
        self._setup_cache()

        if whence == 0:
            self._pos = max(0, min(offset, self._size))
        elif whence == 1:
            self._pos = max(0, min(self._pos + offset, self._size))
        elif whence == 2:
            self._pos = max(0, min(self._size + offset, self._size))
        else:
            raise ValueError("unsupported whence value")

        return self._pos

    @property
    def size(self):
        self._check_closed()
        self._setup_cache()

        return self._size

    def seekable(self):
        self._check_closed()
        return True

    def readable(self):
        self._check_closed()
        return True

    def close(self):
        if not self.closed:
            self._raw.close()
            super(BlockBuffer, self).close()
