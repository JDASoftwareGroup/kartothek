
(Re-)Store a dask index
~~~~~~~~~~~~~~~~~~~~~~~

Calculating a dask index is usually a very expensive operation which requires data to be shuffled around. To (re-)store the dask index we can use the `dask_index_on` keyword.


.. ipython:: python
    :suppress:

    import numpy as np
    import pandas as pd
    from tempfile import TemporaryDirectory

    from kartothek.io.eager import store_dataframes_as_dataset

    dataset_dir = TemporaryDirectory()

    store_url = f"hfs://{dataset_dir.name}"

    df = pd.DataFrame(
        {
            "A": 1.0,
            "B": [
                pd.Timestamp("20130102"),
                pd.Timestamp("20130102"),
                pd.Timestamp("20130103"),
                pd.Timestamp("20130103"),
            ],
            "C": pd.Series(1, index=list(range(4)), dtype="float32"),
            "D": np.array([3] * 4, dtype="int32"),
            "E": pd.Categorical(["test", "train", "test", "train"]),
            "F": "foo",
        }
    )

.. ipython:: python

    import dask.dataframe as dd
    from kartothek.io.dask.dataframe import update_dataset_from_ddf, read_dataset_as_ddf

    df

    ddf = dd.from_pandas(df, npartitions=2)
    ddf_indexed = ddf.set_index("B")

    dm = update_dataset_from_ddf(
        # The key is to reset the index first and let kartothek do the rest
        ddf_indexed.reset_index(),
        table="table",
        dataset_uuid="dataset_ddf_with_index",
        store=store_url,
        partition_on="B",
    ).compute()

    read_dataset_as_ddf(
        dataset_uuid=dm.uuid, store=store_url, dask_index_on="B", table="table"
    )
