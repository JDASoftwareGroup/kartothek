import io

import pytest

from kartothek.serialization._io_buffer import BlockBuffer


class _ReadRecordWrapper:
    """
    Wrapper around file-like objects to record read requests.
    """

    def __init__(self, raw):
        self.raw = raw
        self.records = []

    def __getattr__(self, attr):
        return getattr(self.raw, attr)

    def read(self, size):
        pos = self.raw.tell()
        self.records.append((pos, size))
        return self.raw.read(size)


class _ZeroFile:
    """
    Simulated files filled with NULL-bytes, can be used to test behavior on very large files.
    """

    def __init__(self, size):
        self._size = size
        self._pos = 0

    def tell(self):
        return self._pos

    def seek(self, offset, whence=0):
        if whence == 0:
            self._pos = max(0, min(offset, self._size))
        elif whence == 1:
            self._pos = max(0, min(self._pos + offset, self._size))
        elif whence == 2:
            self._pos = max(0, min(self._size + offset, self._size))
        else:
            raise ValueError("unsupported whence value")

        return self._pos

    def read(self, size=None):
        if (size is None) or (size < 0) or (self._pos + size > self._size):
            # read entire, remaining file
            size = self._size - self._pos

        return b"\0" * size

    def seekable(self):
        return True

    def readable(self):
        return True

    @property
    def closed(self):
        return False


@pytest.fixture
def raw_inner():
    return io.BytesIO(b"foxbar")


@pytest.fixture
def raw(raw_inner):
    return _ReadRecordWrapper(raw_inner)


@pytest.fixture(params=[1, 3, 4, 6, 7, 10])
def blocksize(raw, request):
    return request.param


@pytest.fixture
def example_buffer(raw, blocksize):
    return BlockBuffer(raw, blocksize)


def test_init_fails_not_seekable():
    raw = io.BytesIO()
    raw.seekable = lambda: False
    with pytest.raises(ValueError, match="raw must be seekable"):
        BlockBuffer(raw)


def test_init_fails_not_readable():
    raw = io.BytesIO()
    raw.readable = lambda: False
    with pytest.raises(ValueError, match="raw must be readable"):
        BlockBuffer(raw)


@pytest.mark.parametrize("blocksize", [-1, 0])
def test_init_fails_blocksize(blocksize):
    raw = io.BytesIO()
    with pytest.raises(ValueError, match="blocksize must be at least 1"):
        BlockBuffer(raw, blocksize)


def test_init_fails_closed(blocksize):
    raw = io.BytesIO()
    raw.close()
    with pytest.raises(ValueError, match="Cannot use closed file object"):
        BlockBuffer(raw, blocksize)


def test_size(example_buffer):
    assert example_buffer.size == 6


def test_seekable(example_buffer):
    assert example_buffer.seekable()


def test_readable(example_buffer):
    assert example_buffer.readable()


@pytest.mark.parametrize("size", [None, -1, -100])
def test_read_all(example_buffer, size):
    assert example_buffer.read(size) == b"foxbar"
    assert example_buffer.tell() == 6


def test_read_sequential(example_buffer):
    assert example_buffer.read(2) == b"fo"
    assert example_buffer.tell() == 2
    assert example_buffer.read(3) == b"xba"
    assert example_buffer.tell() == 5
    assert example_buffer.read(0) == b""
    assert example_buffer.tell() == 5
    assert example_buffer.read(2) == b"r"
    assert example_buffer.tell() == 6
    assert example_buffer.read(2) == b""
    assert example_buffer.tell() == 6


def test_fail_illegal_whence(example_buffer):
    with pytest.raises(ValueError, match="unsupported whence value"):
        example_buffer.seek(0, -1)


def test_seek_absolute_and_read(example_buffer):
    assert example_buffer.seek(2) == 2
    assert example_buffer.tell() == 2
    assert example_buffer.read(3) == b"xba"

    assert example_buffer.seek(2) == 2
    assert example_buffer.tell() == 2
    assert example_buffer.read(3) == b"xba"

    # beyond end
    assert example_buffer.seek(100) == 6
    assert example_buffer.tell() == 6
    assert example_buffer.read(3) == b""

    # beyond begin
    assert example_buffer.seek(-1) == 0
    assert example_buffer.tell() == 0
    assert example_buffer.read(1) == b"f"


def test_seek_relative_and_read(example_buffer):
    assert example_buffer.seek(2, 1) == 2
    assert example_buffer.tell() == 2
    assert example_buffer.read(2) == b"xb"

    assert example_buffer.tell() == 4
    assert example_buffer.seek(1, 1) == 5
    assert example_buffer.tell() == 5
    assert example_buffer.read(1) == b"r"

    assert example_buffer.tell() == 6
    assert example_buffer.seek(-3, 1) == 3
    assert example_buffer.tell() == 3
    assert example_buffer.read(1) == b"b"

    assert example_buffer.tell() == 4
    assert example_buffer.seek(0, 1) == 4
    assert example_buffer.tell() == 4
    assert example_buffer.read(1) == b"a"

    # beyond end
    assert example_buffer.tell() == 5
    assert example_buffer.seek(100, 1) == 6
    assert example_buffer.tell() == 6
    assert example_buffer.read(3) == b""

    # beyond begin
    assert example_buffer.seek(-100, 1) == 0
    assert example_buffer.tell() == 0
    assert example_buffer.read(1) == b"f"


def test_seek_reverse_and_read(example_buffer):
    assert example_buffer.seek(-4, 2) == 2
    assert example_buffer.tell() == 2
    assert example_buffer.read(3) == b"xba"

    assert example_buffer.seek(-4, 2) == 2
    assert example_buffer.tell() == 2
    assert example_buffer.read(3) == b"xba"

    # beyond end
    assert example_buffer.seek(1, 2) == 6
    assert example_buffer.tell() == 6
    assert example_buffer.read(3) == b""

    # beyond begin
    assert example_buffer.seek(-10, 2) == 0
    assert example_buffer.tell() == 0
    assert example_buffer.read(1) == b"f"


def test_caching_read_all(raw, example_buffer):
    example_buffer.read()
    example_buffer.read()
    assert raw.records == [(0, 6)]


def test_caching_reuse(raw):
    b = BlockBuffer(raw, 3)

    b.read(1)
    assert raw.records == [(0, 3)]

    b.seek(0)
    b.read(2)
    assert raw.records == [(0, 3)]


def test_caching_remainder(raw):
    b = BlockBuffer(raw, 4)

    b.seek(5)
    b.read()
    assert raw.records == [(4, 2)]


def test_caching_chunks1(raw):
    b = BlockBuffer(raw, 2)

    b.read(1)
    assert raw.records == [(0, 2)]

    b.seek(0)
    b.read()
    assert raw.records == [(0, 2), (2, 4)]


def test_caching_chunks2(raw):
    b = BlockBuffer(raw, 2)

    b.seek(3)
    b.read(1)
    assert raw.records == [(2, 2)]

    b.seek(0)
    b.read()
    assert raw.records == [(2, 2), (0, 2), (4, 2)]


def test_empty(blocksize):
    raw_inner = io.BytesIO()
    raw = _ReadRecordWrapper(raw_inner)
    b = BlockBuffer(raw, blocksize)

    assert b.size == 0
    assert b.read() == b""
    assert raw.records == []


def test_giga():
    raw_size = 100 * 1024 ** 3  # 100 GB
    raw_inner = _ZeroFile(raw_size)
    raw = _ReadRecordWrapper(raw_inner)
    blocksize = 4 * 1024 ** 2  # 4MB
    b = BlockBuffer(raw, blocksize)

    assert b.size == raw_size
    assert b.read(10) == b"\0" * 10
    assert b.seek(-10, 2) == raw_size - 10
    assert b.read(10) == b"\0" * 10
    assert raw.records == [(0, blocksize), (raw_size - blocksize, blocksize)]


def test_closed(blocksize):
    raw = io.BytesIO()
    b = BlockBuffer(raw, blocksize)
    b.close()

    assert b.closed
    assert raw.closed

    with pytest.raises(ValueError, match="I/O operation on closed file."):
        b.size

    with pytest.raises(ValueError, match="I/O operation on closed file."):
        b.seekable()

    with pytest.raises(ValueError, match="I/O operation on closed file."):
        b.readable()

    with pytest.raises(ValueError, match="I/O operation on closed file."):
        b.tell()

    with pytest.raises(ValueError, match="I/O operation on closed file."):
        b.seek(0)

    with pytest.raises(ValueError, match="I/O operation on closed file."):
        b.read()

    # closing twice works
    b.close()


def test_real_file(tmpdir, blocksize):
    path = tmpdir.join("test_real_file.bin").strpath
    with open(path, "wb") as fp:
        fp.write(b"foxbar")

    real_file = open(path, "rb")

    b = BlockBuffer(real_file, blocksize)

    assert not b.closed

    assert b.size == 6
    assert b.seekable() is True
    assert b.readable() is True
    assert b.tell() == 0
    assert b.seek(1) == 1
    assert b.read() == b"oxbar"

    # final close
    b.close()

    # closing twice works
    b.close()
