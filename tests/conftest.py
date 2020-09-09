# noqa: E402
# isort:skip_file

import datetime
import os
import sys
from collections import OrderedDict
from functools import partial

import pandas as pd
import pytest
import storefact

# fmt: off
pytest.register_assert_rewrite("kartothek.io.testing")
# fmt: on

from kartothek.core.common_metadata import make_meta  # noqa: E402
from kartothek.core.factory import DatasetFactory  # noqa: E402
from kartothek.core.testing import (  # noqa: E402
    TIME_TO_FREEZE,
    cm_frozen_time,
    get_dataframe_alltypes,
    get_dataframe_not_nested,
)
from kartothek.io_components.metapartition import (  # noqa: E402
    SINGLE_TABLE,
    MetaPartition,
    gen_uuid,
    parse_input_to_metapartition,
)
from kartothek.core.utils import lazy_store  # noqa: E402
from kartothek.io_components.write import store_dataset_from_partitions  # noqa: E402
from kartothek.serialization import ParquetSerializer  # noqa: E402


pd.options.mode.chained_assignment = "raise"


@pytest.fixture
def frozen_time():
    """
    Depend on this fixture to set the time to TIME_TO_FREEZE
    by patching kartothek.core._time.* with mock objects.

    Note: you only need one of the fixtures `frozen_time`,
    `distributed_frozen_time`, or `frozen_time_em`:

      * if your test function takes a `execution_mode` parameter, use `frozen_time_em`. It will behave
        like `distributed_frozen_time` if `execution_mode` starts with "dask", and like `frozen_time` otherwise.
      * otherwise, if you are testing for dask/distributed, use `distributed_frozen_time`. Note
        that this includes the effects of `frozen_time`.
      * otherwise, use `frozen_time`
    """
    with cm_frozen_time(TIME_TO_FREEZE):
        yield


@pytest.fixture
def frozen_time_em(frozen_time):
    yield


@pytest.fixture
def df_serializer():
    return ParquetSerializer()


@pytest.fixture(scope="session")
def store_session_factory(tmpdir_factory):
    path = tmpdir_factory.mktemp("fsstore_test")
    path = path.realpath()
    url = "hfs://{}".format(path)
    return lazy_store(url)


@pytest.fixture
def store_factory(tmpdir):
    path = tmpdir.join("store").strpath
    url = "hfs://{}".format(path)
    return lazy_store(url)


@pytest.fixture
def store_factory2(tmpdir):
    path = tmpdir.join("store2").strpath
    url = "hfs://{}".format(path)
    return lazy_store(url)


@pytest.fixture(scope="session")
def store_session(store_session_factory):
    return store_session_factory()


@pytest.fixture
def store(store_factory):
    return store_factory()


@pytest.fixture
def store2(store_factory2):
    return store_factory2()


def _check_and_delete(key, store, delete_orig):
    if key not in store.keys():
        raise KeyError(key)
    return delete_orig(key)


def _refuse_write(*args, **kwargs):
    raise RuntimeError("Write operation on read-only store.")


def _get_store(path):
    url = "hfs://{}".format(path)
    store = storefact.get_store_from_url(url)
    store.delete = partial(_check_and_delete, store=store, delete_orig=store.delete)
    return store


def _make_readonly(store):
    if callable(store):
        store = store()

    store.delete = _refuse_write
    store.put = _refuse_write

    return store


@pytest.fixture(scope="function")
def function_store(tmpdir):
    return partial(_get_store, tmpdir.join("function_store").strpath)


@pytest.fixture(scope="function")
def function_store_ro(function_store):
    return partial(_make_readonly, function_store)


@pytest.fixture(scope="function", params=["ro", "rw"])
def function_store_rwro(request, function_store, function_store_ro):
    if request.param == "ro":
        return function_store_ro
    else:
        return function_store


@pytest.fixture(scope="function")
def function_store2(tmpdir):
    return partial(_get_store, tmpdir.join("function_store2").strpath)


@pytest.fixture
def azure_store_cfg_factory(caplog):
    def _f(suffix):
        account_name = "qe72nmjyy6mbi"
        env_name = "AZURE_BLOB_KEY_{}".format(account_name.upper())
        account_key = os.environ.get(env_name)
        if not account_key:
            pytest.skip("{} not provided".format(env_name))

        container = "tests-{component}-{py_major}-{py_minor}-{suffix}".format(
            component="kartothek",
            py_major=sys.version_info.major,
            py_minor=sys.version_info.minor,
            suffix=suffix,
        )

        return {
            "type": "hazure",
            "create_if_missing": True,
            "account_name": account_name,
            "account_key": account_key,
            "container": container,
        }

    return _f


@pytest.fixture(scope="module")
def module_store(tmpdir_factory):
    return partial(_get_store, tmpdir_factory.mktemp("module_store").realpath())


@pytest.fixture
def mock_uuid(mocker):
    uuid = mocker.patch("kartothek.core.uuid._uuid_hook_str")
    uuid.return_value = "auto_dataset_uuid"
    return uuid


@pytest.fixture
def mock_default_metadata_version(mocker, backend_identifier):
    mock_metadata_version = 1

    # Mock `kartothek.core.utils.verify_metadata_version`
    def patched__verify_metadata_version(metadata_version):
        pass

    mocker.patch(
        "kartothek.core.utils._verify_metadata_version",
        patched__verify_metadata_version,
    )

    # Mock `kartothek.io_components.metapartition.parse_input_to_metapartition`
    def patched__parse_input_to_metapartition(
        obj, metadata_version=None, *args, **kwargs
    ):
        if metadata_version == mock_metadata_version:
            table, data = obj  # Tuple
            return MetaPartition(
                label=gen_uuid(), data={table: data}, metadata_version=metadata_version
            )
        try:
            return parse_input_to_metapartition(obj, metadata_version, *args, **kwargs)
        except ValueError as e:
            # Raise a "custom" error to distinguish this error from the error raised
            # by `parse_input_to_metapartition` when the object has not previously
            # passed through this mock function
            raise AssertionError("Traversed through mock. Original error: {}".format(e))

    mocker.patch(
        "kartothek.io.{backend_identifier}.parse_input_to_metapartition".format(
            backend_identifier=backend_identifier
        ),
        patched__parse_input_to_metapartition,
    )

    return mock_metadata_version


@pytest.fixture(params=[4], scope="session")
def metadata_version(request):
    if request.param < 4:
        with pytest.raises(ValueError):
            yield request.param
    else:
        yield request.param


def pytest_runtest_setup(item):
    # If a test is marked with `@pytest.mark.min_metadata_version(x)` the test will
    # be skipped if the `metadata_version` fixture is below this value
    if "metadata_version" in getattr(item, "fixturenames", []):
        minimal_version = item.get_closest_marker("min_metadata_version")
        if minimal_version:
            minimal_version = minimal_version.args[0]
            if minimal_version > item.callspec.params["metadata_version"]:
                pytest.skip("Skipped since the metadata version is too low")


@pytest.fixture(params=["json", "msgpack"], scope="session")
def metadata_storage_format(request):
    return request.param


@pytest.fixture
def df_all_types_schema(df_all_types):
    return make_meta(df_all_types, origin="df_all_types")


@pytest.fixture
def df_all_types_empty_schema(df_all_types):
    df_empty = df_all_types.drop(0)
    assert df_empty.empty
    return make_meta(df_empty, origin="df_empty")


@pytest.fixture
def df_all_types():
    return get_dataframe_alltypes()


@pytest.fixture
def df_not_nested():
    return get_dataframe_not_nested()


def _get_meta_partitions_with_dataframe(metadata_version):
    df = pd.DataFrame(
        OrderedDict(
            [
                ("P", [1]),
                ("L", [1]),
                ("TARGET", [1]),
                ("DATE", [datetime.date(2010, 1, 1)]),
            ]
        )
    )
    df_2 = pd.DataFrame(OrderedDict([("P", [1]), ("info", ["a"])]))
    mp = MetaPartition(
        label="cluster_1",
        data={SINGLE_TABLE: df, "helper": df_2},
        metadata_version=metadata_version,
    )
    df_3 = pd.DataFrame(
        OrderedDict(
            [
                ("P", [2]),
                ("L", [2]),
                ("TARGET", [2]),
                ("DATE", [datetime.date(2009, 12, 31)]),
            ]
        )
    )
    df_4 = pd.DataFrame(OrderedDict([("P", [2]), ("info", ["b"])]))
    mp2 = MetaPartition(
        label="cluster_2",
        data={SINGLE_TABLE: df_3, "helper": df_4},
        metadata_version=metadata_version,
    )
    return [mp, mp2]


@pytest.fixture(scope="session")
def meta_partitions_dataframe(metadata_version):
    """
    Create a list of MetaPartitions for testing. The partitions
    include in-memory pd.DataFrames without external references, i.e. files
     are empty

    """
    with cm_frozen_time(TIME_TO_FREEZE):
        return _get_meta_partitions_with_dataframe(metadata_version)


@pytest.fixture(scope="function")
def meta_partitions_dataframe_function(metadata_version):
    """
    Create a list of MetaPartitions for testing. The partitions
    include in-memory pd.DataFrames without external references, i.e. files
     are empty

    """
    return _get_meta_partitions_with_dataframe(metadata_version)


@pytest.fixture(scope="session")
def meta_partitions_evaluation_dataframe(metadata_version):
    """
    Create a list of MetaPartitions for testing. The partitions
    include in-memory pd.DataFrames without external references, i.e. files
     are empty

    """
    df = pd.DataFrame(
        OrderedDict([("P", [1]), ("L", [1]), ("HORIZON", [1]), ("PRED", [10])])
    )
    mp = MetaPartition(
        label="cluster_1_1", data={"PRED": df}, metadata_version=metadata_version
    )
    df_2 = pd.DataFrame(
        OrderedDict([("P", [1]), ("L", [1]), ("HORIZON", [2]), ("PRED", [20])])
    )
    mp2 = MetaPartition(
        label="cluster_1_2", data={"PRED": df_2}, metadata_version=metadata_version
    )
    df_3 = pd.DataFrame(
        OrderedDict([("P", [2]), ("L", [2]), ("HORIZON", [1]), ("PRED", [10])])
    )
    mp3 = MetaPartition(
        label="cluster_2_1", data={"PRED": df_3}, metadata_version=metadata_version
    )
    df_4 = pd.DataFrame(
        OrderedDict([("P", [2]), ("L", [2]), ("HORIZON", [2]), ("PRED", [20])])
    )
    mp4 = MetaPartition(
        label="cluster_2_2", data={"PRED": df_4}, metadata_version=metadata_version
    )
    return [mp, mp2, mp3, mp4]


def _store_metapartitions(meta_partitions_dataframe, store):
    result = []
    for mp in meta_partitions_dataframe:
        mp = mp.store_dataframes(
            store=store, dataset_uuid="dataset_uuid", store_metadata=True
        )
        result.append(mp)
    return result


@pytest.fixture(scope="session")
def meta_partitions_complete(meta_partitions_dataframe, store_session):
    return _store_metapartitions(meta_partitions_dataframe, store_session)


@pytest.fixture(scope="function")
def meta_partitions_complete_function(meta_partitions_dataframe_function, store):
    return _store_metapartitions(meta_partitions_dataframe_function, store)


@pytest.fixture(scope="session")
def meta_partitions_files_only(meta_partitions_complete):
    """
    Create a list of MetaPartitions for testing. The partitions
    only include external file references with data stored in store

    """
    result = []
    for mp in meta_partitions_complete:
        mp = mp.copy(data={})
        result.append(mp)
    return result


@pytest.fixture(scope="function")
def meta_partitions_files_only_function(meta_partitions_complete_function):
    """
    Create a list of MetaPartitions for testing. The partitions
    only include external file references with data stored in store

    """
    result = []
    for mp in meta_partitions_complete_function:
        mp = mp.copy(data={})
        result.append(mp)
    return result


@pytest.fixture(scope="session")
def meta_partitions_evaluation_files_only(
    meta_partitions_evaluation_dataframe, store_session
):
    """
    Create a list of MetaPartitions for testing. The partitions
    only include external file references with data stored in store

    """
    result = []
    for mp in meta_partitions_evaluation_dataframe:
        mp = mp.store_dataframes(store=store_session, dataset_uuid="evaluation_uuid")
        mp = mp.copy(data={})
        result.append(mp)
    return result


@pytest.fixture(scope="session")
def dataset(meta_partitions_files_only, store_session_factory):
    """
    Create a proper kartothek dataset in store with two partitions

    """
    with cm_frozen_time(TIME_TO_FREEZE):
        return store_dataset_from_partitions(
            partition_list=meta_partitions_files_only,
            dataset_uuid="dataset_uuid",
            store=store_session_factory(),
            dataset_metadata={"dataset": "metadata"},
        )


@pytest.fixture(scope="session")
def dataset_factory(dataset, store_session_factory):
    return DatasetFactory(
        dataset_uuid=dataset.uuid,
        store_factory=store_session_factory,
        load_schema=True,
        load_all_indices=False,
        load_dataset_metadata=True,
    )


@pytest.fixture(scope="session")
def dataset_partition_keys(meta_partitions_dataframe, store_session_factory):
    """
    Create a proper kartothek dataset in store with two partitions

    """
    with cm_frozen_time(TIME_TO_FREEZE):
        new_mps = []
        for mp in meta_partitions_dataframe:
            new_mps.append(mp.partition_on(["P"]))
        new_mps = _store_metapartitions(new_mps, store_session_factory())

        return store_dataset_from_partitions(
            partition_list=new_mps,
            dataset_uuid="dataset_uuid_partition_keys",
            store=store_session_factory(),
            dataset_metadata={"dataset": "metadata"},
        )


@pytest.fixture(scope="session")
def dataset_partition_keys_factory(dataset_partition_keys, store_session_factory):
    return DatasetFactory(
        dataset_uuid=dataset_partition_keys.uuid,
        store_factory=store_session_factory,
        load_schema=True,
        load_all_indices=False,
        load_dataset_metadata=True,
    )


@pytest.fixture(scope="session")
def dataset_with_index(
    meta_partitions_dataframe, store_session_factory, metadata_version
):
    """
    Create a proper kartothek dataset in store with two partitions

    """
    with cm_frozen_time(TIME_TO_FREEZE):
        new_mps = []
        for mp in meta_partitions_dataframe:
            mp = mp.partition_on(["P"])
            mp = mp.build_indices(["L"])
            new_mps.append(mp)

        new_mps = _store_metapartitions(new_mps, store_session_factory())
        return store_dataset_from_partitions(
            partition_list=new_mps,
            dataset_uuid="dataset_uuid_partition_keys_index",
            store=store_session_factory(),
            dataset_metadata={"dataset": "metadata"},
        )


@pytest.fixture(scope="session")
def dataset_with_index_factory(dataset_with_index, store_session_factory):
    return DatasetFactory(
        dataset_uuid=dataset_with_index.uuid,
        store_factory=store_session_factory,
        load_schema=True,
        load_all_indices=False,
        load_dataset_metadata=True,
    )


@pytest.fixture(scope="function")
def dataset_function(meta_partitions_files_only_function, store):
    """
    Create a proper kartothek dataset in store with two partitions

    """
    with cm_frozen_time(TIME_TO_FREEZE):
        return store_dataset_from_partitions(
            partition_list=meta_partitions_files_only_function,
            dataset_uuid="dataset_uuid",
            store=store,
            dataset_metadata={"dataset": "metadata"},
        )


@pytest.fixture(scope="session")
def evaluation_dataset(meta_partitions_evaluation_files_only, store_session):
    with cm_frozen_time(TIME_TO_FREEZE):
        return store_dataset_from_partitions(
            partition_list=meta_partitions_evaluation_files_only,
            dataset_uuid="evaluation_uuid",
            store=store_session,
        )
