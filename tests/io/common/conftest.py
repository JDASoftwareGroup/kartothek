import pytest


@pytest.fixture(params=["eager", "iter", "bag", "delayed", "dataframe"])
def implementation_type(request):
    return request.param


@pytest.fixture(params=["dataframe", "table"])
def output_type(request, implementation_type):
    if (implementation_type in ["iter", "bag", "delayed"]) and (
        request.param == "table"
    ):
        pytest.skip()
    if (implementation_type == "dataframe") and (request.param == "dataframe"):
        pytest.skip()
    return request.param


@pytest.fixture()
def backend_identifier(implementation_type):
    if implementation_type in ["iter", "bag", "delayed"]:
        return f"dask.{implementation_type}"
    else:
        return implementation_type
