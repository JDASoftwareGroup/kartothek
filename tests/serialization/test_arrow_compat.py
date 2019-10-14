# -*- coding: utf-8 -*-


import os
import uuid

import pandas.testing as pdt
import pytest
from storefact import get_store_from_url

from kartothek.core._compat import ARROW_LARGER_EQ_0141
from kartothek.core.testing import get_dataframe_alltypes
from kartothek.serialization import ParquetSerializer


@pytest.fixture(params=["0.12.1", "0.13.0", "0.14.1", "0.15.0"])
def arrow_version(request):
    yield request.param


@pytest.fixture
def reference_store():
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
        "..",
        "reference-data",
        "arrow-compat",
    )
    return get_store_from_url("hfs://{}".format(path))


def test_arrow_compat(arrow_version, reference_store, mocker):
    """
    Test if reading/writing across the supported arrow versions is actually
    compatible

    Generate new reference files with::

        import pyarrow as pa
        ParquetSerializer().store(reference_store, pa.__version__, orig)
    """

    uuid_hook = mocker.patch("kartothek.core.uuid._uuid_hook_object")
    uuid_hook.return_value = uuid.UUID(
        bytes=b"\x82\xd6\xc1\x06Z\x08\x11\xe9\x85eJ\x00\x07\xf8\n\x10"
    )

    orig = get_dataframe_alltypes()
    restored = ParquetSerializer().restore_dataframe(
        store=reference_store, key=arrow_version + ".parquet", date_as_object=True
    )
    if arrow_version == "0.14.1" and not ARROW_LARGER_EQ_0141:
        orig = orig.astype({"null": float})
        pdt.assert_frame_equal(orig, restored)
