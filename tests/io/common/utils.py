import numpy as np
import pandas as pd

from kartothek.io.eager import store_dataframes_as_dataset


def create_dataset(dataset_uuid, store_factory, metadata_version):
    df = pd.DataFrame(
        {"P": np.arange(0, 10), "L": np.arange(0, 10), "TARGET": np.arange(10, 20)}
    )

    df_list = [df.copy(deep=True), df.copy(deep=True)]

    return store_dataframes_as_dataset(
        dfs=df_list,
        store=store_factory,
        dataset_uuid=dataset_uuid,
        metadata_version=metadata_version,
        secondary_indices="P",
    )
