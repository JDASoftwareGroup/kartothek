import pandas as pd
import pytest

from kartothek.io.iter import store_dataframes_as_dataset__iter


@pytest.fixture(params=["eager", "iter", "dask.bag", "dask.delayed", "dask.dataframe"])
def backend_identifier(request):
    return request.param


@pytest.fixture(params=["dataframe", "table"])
def output_type(request, backend_identifier):
    if (backend_identifier in ["iter", "dask.bag", "dask.delayed"]) and (
        request.param == "table"
    ):
        pytest.skip()
    if (backend_identifier == "dask.dataframe") and (request.param == "dataframe"):
        pytest.skip()
    return request.param


@pytest.fixture(scope="session")
def dataset_dispatch_by_uuid():
    import uuid

    return uuid.uuid1().hex


@pytest.fixture(scope="session")
def dataset_dispatch_by(
    metadata_version, store_session_factory, dataset_dispatch_by_uuid
):
    cluster1 = pd.DataFrame(
        {"A": [1, 1], "B": [10, 10], "C": [1, 2], "Content": ["cluster1", "cluster1"]}
    )
    cluster2 = pd.DataFrame(
        {"A": [1, 1], "B": [10, 10], "C": [2, 3], "Content": ["cluster2", "cluster2"]}
    )
    cluster3 = pd.DataFrame({"A": [1], "B": [20], "C": [1], "Content": ["cluster3"]})
    cluster4 = pd.DataFrame(
        {"A": [2, 2], "B": [10, 10], "C": [1, 2], "Content": ["cluster4", "cluster4"]}
    )
    clusters = [cluster1, cluster2, cluster3, cluster4]

    store_dataframes_as_dataset__iter(
        df_generator=clusters,
        store=store_session_factory,
        dataset_uuid=dataset_dispatch_by_uuid,
        metadata_version=metadata_version,
        partition_on=["A", "B"],
        secondary_indices=["C"],
    )
    return pd.concat(clusters).sort_values(["A", "B", "C"]).reset_index(drop=True)
