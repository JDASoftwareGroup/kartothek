Examples
========

Quickstart
^^^^^^^^^^

Setup a store

.. ipython:: python

    from storefact import get_store_from_url
    from functools import partial
    from tempfile import TemporaryDirectory

    # You can, of course, also directly use S3, ABS or anything else
    # supported by :mod:`storefact`
    dataset_dir = TemporaryDirectory()
    store_factory = partial(get_store_from_url, f"hfs://{dataset_dir.name}")


.. ipython:: python

    import pandas as pd
    from kartothek.io.eager import store_dataframes_as_dataset
    from kartothek.io.eager import read_table

    df = pd.DataFrame(
        {'Name': ['Paul', 'Lisa'],
         'Age': [32, 29]}
    )

    dataset_uuid = 'my_list_of_friends'

    metadata = {
        "Name": "My list of friends",
        "Columns": {"Name" : "First name of my friend",
                    "Age": "honest age of my friend in years"}
    }

    store_dataframes_as_dataset(
        store=store_factory,
        dataset_uuid=dataset_uuid,
        dfs=df,
        metadata=metadata
    )

    # Load your data
    # By default the single dataframe is stored in the 'core' table
    df_from_store = read_table(
        store=store_factory,
        dataset_uuid=dataset_uuid,
        table='table'
    )
    df_from_store


Write
^^^^^

.. ipython:: python

    import pandas as pd
    from kartothek.io.eager import store_dataframes_as_dataset

    # Now, define the actual partitions. This list will, most of the time,
    # be the intermediate result of a previously executed pipeline which e.g. pulls
    # data from an external data source
    # In our particular case, we'll use manual input and define our partitions explicitly

    # We'll define two partitions which both have two tables
    input_list_of_partitions = [
        {
            "label": "FirstPartition",
            "data": [
                ("FirstCategory", pd.DataFrame()),
                ("SecondCategory", pd.DataFrame())
            ]
        },
        {
            "label": "SecondPartition",
            "data": [
                ("FirstCategory", pd.DataFrame()),
                ("SecondCategory", pd.DataFrame())
            ]
        }
    ]

    # The pipeline will return a :class:`~kartothek.core.dataset.DatasetMetadata` object
    # which refers to the created dataset
    dataset = store_dataframes_as_dataset(
        dfs=input_list_of_partitions,
        store=store_factory,
        dataset_uuid='MyFirstDataset',
        metadata={"dataset": "metadata"},  # This is optional dataset metadata
        metadata_version=4
    )
    dataset


Read
^^^^

.. ipython:: python

    import pandas as pd
    from kartothek.io.iter import read_dataset_as_dataframes__iterator

    # Create the pipeline with a minimal set of configs
    list_of_partitions = read_dataset_as_dataframes__iterator(
        dataset_uuid='MyFirstDataset',
        store=store_factory
    )
    # the iter backend returns a generator object. In our case we want to look at
    # all partitions at once
    list_of_partitions = list(list_of_partitions)

    # In case you were using the dataset created in the Write example
    for d1, d2 in zip(list_of_partitions, [
                            # FirstPartition
                            {
                                "FirstCategory": pd.DataFrame(),
                                "SecondCategory": pd.DataFrame()
                            },
                            # SecondPartition
                            {
                                "FirstCategory": pd.DataFrame(),
                                "SecondCategory": pd.DataFrame()
                            },
                        ]
    ):
        for kv1, kv2 in zip(d1.items(), d2.items()):
            k1, v1 = kv1
            k2, v2 = kv2
            assert k1 == k2 and all(v1 == v2)
