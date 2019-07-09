# -*- coding: utf-8 -*-
import pickle

import pytest

from kartothek.io.dask.bag import build_dataset_indices__bag
from kartothek.io.testing.index import *  # noqa: F4


def _build_indices(*args, **kwargs):
    bag = build_dataset_indices__bag(*args, **kwargs)

    # pickle roundtrip to ensure we don't need the inefficient cloudpickle fallback
    s = pickle.dumps(bag, pickle.HIGHEST_PROTOCOL)
    bag = pickle.loads(s)

    bag.compute()


@pytest.fixture()
def bound_build_dataset_indices():
    return _build_indices
