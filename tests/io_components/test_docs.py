import inspect

import pytest

from kartothek.io.dask.bag import (
    build_dataset_indices__bag,
    read_dataset_as_dataframe_bag,
    read_dataset_as_metapartitions_bag,
    store_bag_as_dataset,
)
from kartothek.io.dask.dataframe import read_dataset_as_ddf, update_dataset_from_ddf
from kartothek.io.dask.delayed import (
    delete_dataset__delayed,
    merge_datasets_as_delayed,
    read_dataset_as_delayed,
    read_dataset_as_delayed_metapartitions,
    read_table_as_delayed,
    store_delayed_as_dataset,
    update_dataset_from_delayed,
)
from kartothek.io.eager import store_dataframes_as_dataset
from kartothek.io_components.docs import _PARAMETER_MAPPING


@pytest.mark.parametrize(
    "function",
    [
        read_dataset_as_metapartitions_bag,
        read_dataset_as_dataframe_bag,
        store_bag_as_dataset,
        build_dataset_indices__bag,
        read_dataset_as_ddf,
        update_dataset_from_ddf,
        delete_dataset__delayed,
        merge_datasets_as_delayed,
        read_dataset_as_delayed_metapartitions,
        read_dataset_as_delayed,
        read_table_as_delayed,
        update_dataset_from_delayed,
        store_delayed_as_dataset,
    ],
)
def test_docs(function):
    docstrings = store_dataframes_as_dataset.__doc__
    arguments = inspect.signature(store_dataframes_as_dataset).parameters
    assert all(
        [
            _PARAMETER_MAPPING.get(argument, "Parameters") in docstrings
            for argument in arguments
        ]
    )
