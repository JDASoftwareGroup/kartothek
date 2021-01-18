"""
Common functionality required to implement cube write functionality.
"""
import itertools
from copy import copy
from typing import Dict, Iterable, Optional, Sequence, Tuple

import dask.dataframe as dd
import pandas as pd
from pandas.api.types import is_sparse

from kartothek.api.consistency import check_datasets, get_payload_subset
from kartothek.core.common_metadata import store_schema_metadata
from kartothek.core.cube.constants import (
    KTK_CUBE_METADATA_DIMENSION_COLUMNS,
    KTK_CUBE_METADATA_KEY_IS_SEED,
    KTK_CUBE_METADATA_PARTITION_COLUMNS,
    KTK_CUBE_METADATA_SUPPRESS_INDEX_ON,
    KTK_CUBE_METADATA_VERSION,
)
from kartothek.core.cube.cube import Cube
from kartothek.core.dataset import DatasetMetadataBuilder
from kartothek.core.naming import metadata_key_from_uuid
from kartothek.core.uuid import gen_uuid
from kartothek.io_components.metapartition import SINGLE_TABLE, MetaPartition
from kartothek.utils.converters import converter_str
from kartothek.utils.pandas import mask_sorted_duplicates_keep_last, sort_dataframe

__all__ = (
    "apply_postwrite_checks",
    "check_datasets_prebuild",
    "check_datasets_preextend",
    "check_provided_metadata_dict",
    "multiplex_user_input",
    "prepare_data_for_ktk",
    "prepare_ktk_metadata",
    "prepare_ktk_partition_on",
)


def check_provided_metadata_dict(metadata, ktk_cube_dataset_ids):
    """
    Check metadata dict provided by the user.

    Parameters
    ----------
    metadata: Optional[Dict[str, Dict[str, Any]]]
        Optional metadata provided by the user.
    ktk_cube_dataset_ids: Iterable[str]
        ktk_cube_dataset_ids announced by the user.

    Returns
    -------
    metadata: Dict[str, Dict[str, Any]]
        Metadata provided by the user.

    Raises
    ------
    TypeError: If either the dict or one of the contained values has the wrong type.
    ValueError: If a ktk_cube_dataset_id in the dict is not in ktk_cube_dataset_ids.
    """
    if metadata is None:
        metadata = {}
    elif not isinstance(metadata, dict):
        raise TypeError(
            "Provided metadata should be a dict but is {}".format(
                type(metadata).__name__
            )
        )

    unknown_ids = set(metadata.keys()) - set(ktk_cube_dataset_ids)
    if unknown_ids:
        raise ValueError(
            "Provided metadata for otherwise unspecified ktk_cube_dataset_ids: {}".format(
                ", ".join(sorted(unknown_ids))
            )
        )

    # sorted iteration for deterministic error messages
    for k in sorted(metadata.keys()):
        v = metadata[k]
        if not isinstance(v, dict):
            raise TypeError(
                "Provided metadata for dataset {} should be a dict but is {}".format(
                    k, type(v).__name__
                )
            )

    return metadata


def prepare_ktk_metadata(cube, ktk_cube_dataset_id, metadata):
    """
    Prepare metadata that should be passed to Kartothek.

    This will add the following information:

    - a flag indicating whether the dataset is considered a seed dataset
    - dimension columns
    - partition columns
    - optional user-provided metadata

    Parameters
    ----------
    cube: kartothek.core.cube.cube.Cube
        Cube specification.
    ktk_cube_dataset_id: str
        Ktk_cube dataset UUID (w/o cube prefix).
    metadata: Optional[Dict[str, Dict[str, Any]]]
        Optional metadata provided by the user. The first key is the ktk_cube dataset id,
        the value is the user-level metadata for that dataset. Should be piped through
        :meth:`check_provided_metadata_dict` beforehand.

    Returns
    -------
    ktk_metadata: Dict[str, Any]
        Metadata ready for Kartothek.
    """
    if metadata is None:
        metadata = {}

    ds_metadata = metadata.get(ktk_cube_dataset_id, {})
    ds_metadata[KTK_CUBE_METADATA_DIMENSION_COLUMNS] = list(cube.dimension_columns)
    ds_metadata[KTK_CUBE_METADATA_KEY_IS_SEED] = (
        ktk_cube_dataset_id == cube.seed_dataset
    )
    ds_metadata[KTK_CUBE_METADATA_PARTITION_COLUMNS] = list(cube.partition_columns)
    ds_metadata[KTK_CUBE_METADATA_SUPPRESS_INDEX_ON] = list(cube.suppress_index_on)

    return ds_metadata


def assert_dimesion_index_cols_notnull(
    df: pd.DataFrame, ktk_cube_dataset_id: str, cube: Cube, partition_on: Sequence[str]
) -> pd.DataFrame:
    """
    Assert that index and dimesion columns are not NULL and raise an appropriate error if so.

    .. note::

        Indices for plain non-cube dataset drop null during index build!
    """

    df_columns_set = set(df.columns)
    dcols_present = set(cube.dimension_columns) & df_columns_set
    icols_present = set(cube.index_columns) & df_columns_set

    for cols, what in (
        (partition_on, "partition"),
        (dcols_present, "dimension"),
        (icols_present, "index"),
    ):
        for col in sorted(cols):
            if df[col].isnull().any():
                raise ValueError(
                    'Found NULL-values in {what} column "{col}" of dataset "{ktk_cube_dataset_id}"'.format(
                        col=col, ktk_cube_dataset_id=ktk_cube_dataset_id, what=what
                    )
                )
    return df


def check_user_df(ktk_cube_dataset_id, df, cube, existing_payload, partition_on):
    """
    Check user-provided DataFrame for sanity.

    Parameters
    ----------
    ktk_cube_dataset_id: str
        Ktk_cube dataset UUID (w/o cube prefix).
    df: Optional[pandas.DataFrame]
        DataFrame to be passed to Kartothek.
    cube: kartothek.core.cube.cube.Cube
        Cube specification.
    existing_payload: Set[str]
        Existing payload columns.
    partition_on: Iterable[str]
        Partition-on attribute for given dataset.

    Raises
    ------
    ValueError
        In case anything is fishy.
    """
    if df is None:
        return
    if not (isinstance(df, pd.DataFrame) or isinstance(df, dd.DataFrame)):
        raise TypeError(
            'Provided DataFrame is not a pandas.DataFrame or None, but is a "{t}"'.format(
                t=type(df).__name__
            )
        )
    if any(is_sparse(dtype) for dtype in df.dtypes):
        raise TypeError("Sparse data is not supported.")

    # call this once since `df.columns` can be quite slow
    df_columns = list(df.columns)
    df_columns_set = set(df_columns)
    dcols_present = set(cube.dimension_columns) & df_columns_set

    if len(df_columns) != len(df_columns_set):
        raise ValueError(
            'Duplicate columns found in dataset "{ktk_cube_dataset_id}": {df_columns}'.format(
                ktk_cube_dataset_id=ktk_cube_dataset_id,
                df_columns=", ".join(df_columns),
            )
        )

    if ktk_cube_dataset_id == cube.seed_dataset:
        missing_dimension_columns = set(cube.dimension_columns) - df_columns_set
        if missing_dimension_columns:
            raise ValueError(
                'Missing dimension columns in seed data "{ktk_cube_dataset_id}": {missing_dimension_columns}'.format(
                    ktk_cube_dataset_id=ktk_cube_dataset_id,
                    missing_dimension_columns=", ".join(
                        sorted(missing_dimension_columns)
                    ),
                )
            )
    else:
        if len(dcols_present) == 0:
            raise ValueError(
                'Dataset "{ktk_cube_dataset_id}" must have at least 1 of the following dimension columns: {dims}'.format(
                    ktk_cube_dataset_id=ktk_cube_dataset_id,
                    dims=", ".join(cube.dimension_columns),
                )
            )

    missing_partition_columns = set(partition_on) - df_columns_set
    if missing_partition_columns:
        raise ValueError(
            'Missing partition columns in dataset "{ktk_cube_dataset_id}": {missing_partition_columns}'.format(
                ktk_cube_dataset_id=ktk_cube_dataset_id,
                missing_partition_columns=", ".join(sorted(missing_partition_columns)),
            )
        )

    # Factor this check out. All others can be performed on the dask.DataFrame.
    # This one can only be executed on a pandas DataFame
    if isinstance(df, pd.DataFrame):
        assert_dimesion_index_cols_notnull(
            ktk_cube_dataset_id=ktk_cube_dataset_id,
            df=df,
            cube=cube,
            partition_on=partition_on,
        )

    payload = get_payload_subset(df.columns, cube)
    payload_overlap = payload & existing_payload
    if payload_overlap:
        raise ValueError(
            'Payload written in "{ktk_cube_dataset_id}" is already present in cube: {payload_overlap}'.format(
                ktk_cube_dataset_id=ktk_cube_dataset_id,
                payload_overlap=", ".join(sorted(payload_overlap)),
            )
        )

    unspecified_partition_columns = (df_columns_set - set(partition_on)) & set(
        cube.partition_columns
    )
    if unspecified_partition_columns:
        raise ValueError(
            f"Unspecified but provided partition columns in {ktk_cube_dataset_id}: "
            f"{', '.join(sorted(unspecified_partition_columns))}"
        )


def _check_duplicates(ktk_cube_dataset_id, df, sort_keys, cube):
    dup_mask = mask_sorted_duplicates_keep_last(df, sort_keys)
    if dup_mask.any():
        df_with_dups = df.iloc[dup_mask]
        example_row = df_with_dups.iloc[0]

        df_dup = df.loc[(df.loc[:, sort_keys] == example_row[sort_keys]).all(axis=1)]

        cols_id = set(df_dup.columns[df_dup.nunique() == 1])
        cols_show_id = cols_id - set(sort_keys)
        cols_show_nonid = set(df.columns) - cols_id
        raise ValueError(
            f'Found duplicate cells by [{", ".join(sorted(sort_keys))}] in dataset "{ktk_cube_dataset_id}", example:\n'
            f"\n"
            f"Keys:\n"
            f"{example_row[sorted(sort_keys)].to_string()}\n"
            f"\n"
            f"Identical Payload:\n"
            f'{example_row[sorted(cols_show_id)].to_string() if cols_show_id else "n/a"}\n'
            f"\n"
            f'Non-Idential Payload:\n{df_dup[sorted(cols_show_nonid)].to_string() if cols_show_nonid else "n/a"}'
        )


def prepare_data_for_ktk(
    df, ktk_cube_dataset_id, cube, existing_payload, partition_on, consume_df=False
):
    """
    Prepare data so it can be handed over to Kartothek.

    Some checks will be applied to the data to ensure it is sane.

    Parameters
    ----------
    df: pandas.DataFrame
        DataFrame to be passed to Kartothek.
    ktk_cube_dataset_id: str
        Ktk_cube dataset UUID (w/o cube prefix).
    cube: kartothek.core.cube.cube.Cube
        Cube specification.
    existing_payload: Set[str]
        Existing payload columns.
    partition_on: Iterable[str]
        Partition-on attribute for given dataset.
    consume_df: bool
        Whether the incoming DataFrame can be destroyed while processing it.

    Returns
    -------
    mp: Kartothek.io.metapartition.MetaPartition
        Kartothek-ready MetaPartition, may be sentinel (aka empty and w/o label).

    Raises
    ------
    ValueError
        In case anything is fishy.
    """
    check_user_df(ktk_cube_dataset_id, df, cube, existing_payload, partition_on)

    if (df is None) or df.empty:
        # fast-path for empty DF
        return MetaPartition(
            label=None,
            metadata_version=KTK_CUBE_METADATA_VERSION,
            partition_keys=list(partition_on),
        )

    # TODO: find a more elegant solution that works w/o copy
    df_orig = df
    df = df.copy()
    if consume_df:
        # the original df is still referenced in the parent scope, so drop it
        df_orig.drop(columns=df_orig.columns, index=df_orig.index, inplace=True)
    df_columns = list(df.columns)
    df_columns_set = set(df_columns)

    # normalize value order and reset index
    sort_keys = [
        col
        for col in itertools.chain(cube.partition_columns, cube.dimension_columns)
        if col in df_columns_set
    ]
    df = sort_dataframe(df=df, columns=sort_keys)

    # check duplicate cells
    _check_duplicates(ktk_cube_dataset_id, df, sort_keys, cube)

    # check+convert column names to unicode strings
    df.rename(columns={c: converter_str(c) for c in df_columns}, inplace=True)

    # create MetaPartition object for easier handling
    mp = MetaPartition(
        label=gen_uuid(),
        data={SINGLE_TABLE: df},
        metadata_version=KTK_CUBE_METADATA_VERSION,
    )
    del df

    # partition data
    mp = mp.partition_on(list(partition_on))

    # reset indices again (because partition_on breaks it)
    for mp2 in mp.metapartitions:
        mp2["data"][SINGLE_TABLE].reset_index(drop=True, inplace=True)
        del mp2

    # calculate indices
    indices_to_build = set(cube.index_columns) & df_columns_set
    if ktk_cube_dataset_id == cube.seed_dataset:
        indices_to_build |= set(cube.dimension_columns) - set(cube.suppress_index_on)
    indices_to_build -= set(partition_on)

    mp = mp.build_indices(indices_to_build)

    return mp


def multiplex_user_input(data, cube):
    """
    Get input from the user and ensure it's a multi-dataset dict.

    Parameters
    ----------
    data: Union[pandas.DataFrame, Dict[str, pandas.DataFrame]]
        User input.
    cube: kartothek.core.cube.cube.Cube
        Cube specification.

    Returns
    -------
    pipeline_input: Dict[str, pandas.DataFrame]
        Input for write pipelines.
    """
    if not isinstance(data, dict):
        data = {cube.seed_dataset: data}
    return data


class MultiTableCommitAborted(RuntimeError):
    """An Error occured during the commit of a MultiTable dataset (Cube) causing a rollback."""


def apply_postwrite_checks(datasets, cube, store, existing_datasets):
    """
    Apply sanity checks that can only be done after Kartothek has written its datasets.

    Parameters
    ----------
    datasets: Dict[str, kartothek.core.dataset.DatasetMetadata]
        Datasets that just got written.
    cube: kartothek.core.cube.cube.Cube
        Cube specification.
    store: Union[Callable[[], simplekv.KeyValueStore], simplekv.KeyValueStore]
        KV store.
    existing_datasets: Dict[str, kartothek.core.dataset.DatasetMetadata]
        Datasets that were present before the write procedure started.

    Returns
    -------
    datasets: Dict[str, kartothek.core.dataset.DatasetMetadata]
        Datasets that just got written.

    Raises
    ------
    ValueError
        If sanity check failed.
    """
    try:
        empty_datasets = {
            ktk_cube_dataset_id
            for ktk_cube_dataset_id, ds in datasets.items()
            if SINGLE_TABLE not in ds.table_meta or len(ds.partitions) == 0
        }

        if empty_datasets:
            raise ValueError(
                "Cannot write empty datasets: {empty_datasets}".format(
                    empty_datasets=", ".join(sorted(empty_datasets))
                )
            )

        datasets_to_check = copy(existing_datasets)
        datasets_to_check.update(datasets)
        check_datasets(datasets_to_check, cube)
    except Exception as e:
        _rollback_transaction(
            existing_datasets=existing_datasets, new_datasets=datasets, store=store
        )

        raise MultiTableCommitAborted(
            "Post commit check failed. Operation rolled back."
        ) from e

    return datasets


def check_datasets_prebuild(ktk_cube_dataset_ids, cube, existing_datasets):
    """
    Check if given dataset UUIDs can be used to build a given cube, to be used before any write operation is performed.

    The following checks will be applied:

    - the seed dataset must be part of the data
    - no leftovers (non-seed datasets) must be present that are not overwritten

    Parameters
    ----------
    ktk_cube_dataset_ids: Iterable[str]
        Dataset IDs that should be written.
    cube: kartothek.core.cube.cube.Cube
        Cube specification.
    existing_datasets: Dict[str, kartothek.core.dataset.DatasetMetadata]
        Datasets that existings before the write process started.

    Raises
    ------
    ValueError
        In case of an error.
    """
    if cube.seed_dataset not in ktk_cube_dataset_ids:
        raise ValueError('Seed data ("{}") is missing.'.format(cube.seed_dataset))

    missing_overwrites = set(existing_datasets.keys()) - set(ktk_cube_dataset_ids)
    if missing_overwrites:
        raise ValueError(
            "Following datasets exists but are not overwritten (partial overwrite), this is not allowed: {}".format(
                ", ".join(sorted(missing_overwrites))
            )
        )


def check_datasets_preextend(ktk_cube_dataset_ids, cube):
    """
    Check if given dataset UUIDs can be used to extend a given cube, to be used before any write operation is performed.

    The following checks will be applied:

    - the seed dataset of the cube must not be touched

    ..warning::
        It is assumed that Kartothek checks if the ``overwrite`` flags are correct. Therefore, modifications of non-seed
        datasets are NOT checked here.

    Parameters
    ----------
    ktk_cube_dataset_ids: Iterable[str]
        Dataset IDs that should be written.
    cube: kartothek.core.cube.cube.Cube
        Cube specification.

    Raises
    ------
    ValueError
        In case of an error.
    """
    if cube.seed_dataset in ktk_cube_dataset_ids:
        raise ValueError(
            'Seed data ("{}") cannot be written during extension.'.format(
                cube.seed_dataset
            )
        )


def _rollback_transaction(existing_datasets, new_datasets, store):
    """
    Rollback changes made during tht write process.

    Parameters
    ----------
    existing_datasets: Dict[str, kartothek.core.dataset.DatasetMetadata]
        Datasets that existings before the write process started.
    new_datasets: Dict[str, kartothek.core.dataset.DatasetMetadata]
        Datasets that where created / changed during the write process.
    store: Union[Callable[[], simplekv.KeyValueStore], simplekv.KeyValueStore]
        KV store.
    """
    if callable(store):
        store = store()

    # delete newly created datasets that where not present before the "transaction"
    for ktk_cube_dataset_id in sorted(set(new_datasets) - set(existing_datasets)):
        store.delete(metadata_key_from_uuid(new_datasets[ktk_cube_dataset_id].uuid))

    # recover changes of old datasets
    for ktk_cube_dataset_id in sorted(set(new_datasets) & set(existing_datasets)):
        ds = existing_datasets[ktk_cube_dataset_id]
        builder = DatasetMetadataBuilder.from_dataset(ds)
        store.put(*builder.to_json())
        for table, schema in ds.table_meta.items():
            store_schema_metadata(
                schema=schema, dataset_uuid=ds.uuid, store=store, table=table
            )


def prepare_ktk_partition_on(
    cube: Cube,
    ktk_cube_dataset_ids: Iterable[str],
    partition_on: Optional[Dict[str, Iterable[str]]],
) -> Dict[str, Tuple[str, ...]]:
    """
    Prepare ``partition_on`` values for kartothek.

    Parameters
    ----------
    cube:
        Cube specification.
    ktk_cube_dataset_ids:
        ktk_cube_dataset_ids announced by the user.
    partition_on:
        Optional parition-on attributes for datasets.

    Returns
    -------
    partition_on:
        Partition-on per dataset.

    Raises
    ------
    ValueError: In case user-provided values are invalid.
    """
    if partition_on is None:
        partition_on = {}

    default = cube.partition_columns

    result = {}
    for ktk_cube_dataset_id in ktk_cube_dataset_ids:
        po = tuple(partition_on.get(ktk_cube_dataset_id, default))

        if ktk_cube_dataset_id == cube.seed_dataset:
            if po != default:
                raise ValueError(
                    f"Seed dataset {ktk_cube_dataset_id} must have the following, fixed partition-on attribute: "
                    f"{', '.join(default)}"
                )
        if len(set(po)) != len(po):
            raise ValueError(
                f"partition-on attribute of dataset {ktk_cube_dataset_id} contains duplicates: {', '.join(po)}"
            )

        result[ktk_cube_dataset_id] = po

    return result
