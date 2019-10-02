import string

import numpy as np
import pandas as pd
import storefact
from simplekv.decorator import StoreDecorator

from kartothek.io.eager import store_dataframes_as_dataset
from kartothek.io_components.metapartition import SINGLE_TABLE


def create_dataset(dataset_uuid, store_factory, metadata_version):
    df = pd.DataFrame(
        {"P": np.arange(0, 10), "L": np.arange(0, 10), "TARGET": np.arange(10, 20)}
    )

    df_helper = pd.DataFrame(
        {"P": np.arange(0, 10), "info": string.ascii_lowercase[:10]}
    )

    df_list = [
        {
            "label": "cluster_1",
            "data": [(SINGLE_TABLE, df.copy(deep=True)), ("helper", df_helper)],
            "indices": {"P": {val: ["cluster_2"] for val in df.TARGET.unique()}},
        },
        {
            "label": "cluster_2",
            "data": [(SINGLE_TABLE, df.copy(deep=True)), ("helper", df_helper)],
            "indices": {"P": {val: ["cluster_2"] for val in df.TARGET.unique()}},
        },
    ]

    return store_dataframes_as_dataset(
        dfs=df_list,
        store=store_factory,
        dataset_uuid=dataset_uuid,
        metadata_version=metadata_version,
    )


class NoPickleDecorator(StoreDecorator):
    def __getstate__(self):
        raise RuntimeError("do NOT pickle this object!")


def no_pickle_store_from_url(url):
    store = storefact.get_store_from_url(url)
    return NoPickleDecorator(store)
