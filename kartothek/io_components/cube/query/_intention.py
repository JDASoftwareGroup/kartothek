"""
Methods to figure out the actual user intention during the query process.
"""
import itertools
import typing
from functools import reduce

import attr
import pandas as pd
import pyarrow as pa

from kartothek.core.cube.conditions import Conjunction
from kartothek.serialization._parquet import _normalize_value
from kartothek.utils.converters import converter_str_set, converter_str_tupleset
from kartothek.utils.ktk_adapters import get_dataset_columns, get_dataset_schema

__all__ = ("QueryIntention", "determine_intention")


def _process_dimension_columns(dimension_columns, cube):
    """
    Process and check given dimension columns.

    Parameters
    ----------
    dimension_columns: Optional[Iterable[str]]
        Dimension columns of the query, may result in projection.
    cube: Cube
        Cube specification.

    Returns
    -------
    dimension_columns: Tuple[str, ...]
        Real dimension columns.
    """
    if dimension_columns is None:
        return cube.dimension_columns
    else:
        dimension_columns = converter_str_tupleset(dimension_columns)
        missing = set(dimension_columns) - set(cube.dimension_columns)
        if missing:
            raise ValueError(
                "Following dimension columns were requested but are missing from the cube: {missing}".format(
                    missing=", ".join(sorted(missing))
                )
            )
        if len(dimension_columns) == 0:
            raise ValueError("Dimension columns cannot be empty.")
        return dimension_columns


def _process_partition_by(partition_by, cube, all_available_columns, indexed_columns):
    """
    Process and check given partition-by columns.

    Parameters
    ----------
    partition_by: Optional[Iterable[str]]
        By which column logical partitions should be formed.
    cube: Cube
        Cube specification.
    all_available_columns: Set[str]
        All columns that are available for query.
    indexed_columns: Dict[str, Set[str]]
        Indexed columns per ktk_cube dataset ID.

    Returns
    -------
    partition_by: Tuple[str, ...]
        Real partition-by columns, may be empty.
    """
    if partition_by is None:
        return []
    else:
        partition_by = converter_str_tupleset(partition_by)
        partition_by_set = set(partition_by)

        missing_available = partition_by_set - all_available_columns
        if missing_available:
            raise ValueError(
                "Following partition-by columns were requested but are missing from the cube: {missing}".format(
                    missing=", ".join(sorted(missing_available))
                )
            )

        missing_indexed = partition_by_set - reduce(
            set.union, indexed_columns.values(), set()
        )
        if missing_indexed:
            raise ValueError(
                "Following partition-by columns are not indexed and cannot be used: {missing}".format(
                    missing=", ".join(sorted(missing_indexed))
                )
            )

        return partition_by


def _test_condition_types(conditions, datasets):
    """
    Process and check given query conditions.

    Parameters
    ----------
    conditions: Conjunction
        Conditions that should be applied.
    datasets: Dict[str, kartothek.core.dataset.DatasetMetadata]
        Datasets that are present.

    Raises
    -------
    TypeError: In case of a wrong type.
    """
    for single_condition in conditions.conditions:
        test_predicate = single_condition.predicate_part
        for literal in test_predicate:
            col, op, val = literal
            if op != "in":
                val = [val]

            for ktk_cube_dataset_id in sorted(datasets.keys()):
                dataset = datasets[ktk_cube_dataset_id]
                meta = get_dataset_schema(dataset)
                if col not in meta.names:
                    continue
                pa_type = meta.field(col).type

                if pa.types.is_null(pa_type):
                    # ignore all-NULL columns
                    # TODO: the query planner / regrouper could use that to emit 0 partitions
                    continue

                for v in val:
                    try:
                        _normalize_value(v, pa_type)

                        # special check for numpy signed vs unsigned integers
                        if hasattr(v, "dtype"):
                            dtype = v.dtype
                            if (
                                pd.api.types.is_unsigned_integer_dtype(dtype)
                                and pa.types.is_signed_integer(pa_type)
                            ) or (
                                pd.api.types.is_signed_integer_dtype(dtype)
                                and pa.types.is_unsigned_integer(pa_type)
                            ):
                                # proper exception message will be constructed below
                                raise TypeError()
                    except Exception:
                        raise TypeError(
                            (
                                "Condition `{single_condition}` has wrong type. Expected `{pa_type} ({pd_type})` but "
                                "got `{value} ({py_type})`"
                            ).format(
                                single_condition=single_condition,
                                pa_type=pa_type,
                                pd_type=pa_type.to_pandas_dtype().__name__,
                                value=v,
                                py_type=type(v).__name__,
                            )
                        )


def _process_conditions(
    conditions, cube, datasets, all_available_columns, indexed_columns
):
    """
    Process and check given query conditions.

    Parameters
    ----------
    conditions: Union[None, Condition, Iterable[Condition], Conjunction]
        Conditions that should be applied.
    cube: Cube
        Cube specification.
    datasets: Dict[str, kartothek.core.dataset.DatasetMetadata]
        Datasets that are present.
    all_available_columns: Set[str]
        All columns that are available for query.
    indexed_columns: Dict[str, Set[str]]
        Indexed columns per ktk_cube dataset ID.

    Returns
    -------
    conditions_pre: Dict[str, Conjuction]
        Conditions to be applied based on the index data alone.
    conditions_post: Dict[str, Conjuction]
        Conditions to be applied during the load process.

    Raises
    -------
    TypeError: In case of a wrong type.
    """
    conditions = Conjunction(conditions)

    condition_columns = conditions.columns
    missing = condition_columns - all_available_columns
    if missing:
        raise ValueError(
            "Following condition columns are required but are missing from the cube: {missing}".format(
                missing=", ".join(sorted(missing))
            )
        )
    _test_condition_types(conditions, datasets)

    conditions_split = conditions.split_by_column()

    conditions_pre = {}
    for ktk_cube_dataset_id, ds in datasets.items():
        candidate_cols = indexed_columns[ktk_cube_dataset_id]
        if not candidate_cols:
            continue

        filtered = [
            conj for col, conj in conditions_split.items() if col in candidate_cols
        ]
        if not filtered:
            continue

        conditions_pre[ktk_cube_dataset_id] = reduce(Conjunction.from_two, filtered)

    conditions_post = {}
    for ktk_cube_dataset_id, ds in datasets.items():
        candidate_cols = (get_dataset_columns(ds) & condition_columns) - set(
            cube.partition_columns
        )
        if not candidate_cols:
            continue

        filtered = [
            conj for col, conj in conditions_split.items() if col in candidate_cols
        ]
        if not filtered:
            continue

        conditions_post[ktk_cube_dataset_id] = reduce(Conjunction.from_two, filtered)

    return conditions_pre, conditions_post


def _process_payload(payload_columns, all_available_columns, cube):
    """
    Process and check given payload columns.

    Parameters
    ----------
    payload_columns: Optional[Iterable[str]]
        Which columns apart from ``dimension_columns`` and ``partition_by`` should be returned from the query.
    all_available_columns: Set[str]
        All columns that are available for query.
    cube: Cube
        Cube specification.

    Returns
    -------
    payload_columns: Set[str]
        Payload columns to be returned from the query.
    """
    if payload_columns is None:
        return all_available_columns
    else:
        payload_columns = converter_str_set(payload_columns)
        missing = payload_columns - all_available_columns
        if missing:
            raise ValueError(
                "Cannot find the following requested payload columns: {missing}".format(
                    missing=", ".join(sorted(missing))
                )
            )
        return payload_columns


def determine_intention(
    cube,
    datasets,
    dimension_columns,
    partition_by,
    conditions,
    payload_columns,
    indexed_columns,
):
    """
    Dermine and check user intention during the query process.

    Parameters
    ----------
    cube: Cube
        Cube specification.
    datasets: Dict[str, kartothek.core.dataset.DatasetMetadata]
        Datasets that are present.
    dimension_columns: Optional[Iterable[str]]
        Dimension columns of the query, may result in projection.
    partition_by: Optional[Iterable[str]]
        By which column logical partitions should be formed.
    conditions: Union[None, Condition, Iterable[Condition], Conjunction]
        Conditions that should be applied.
    payload_columns: Optional[Iterable[str]]
        Which columns apart from ``dimension_columns`` and ``partition_by`` should be returned from the query.
    indexed_columns: Dict[str, Set[str]]
        Indexed columns per ktk_cube dataset ID.

    Returns
    -------
    intention: QueryIntention
        Checked and filled in intention of the user.
    """
    all_available_columns = set(
        itertools.chain.from_iterable(
            [get_dataset_columns(ds) for ds in datasets.values()]
        )
    )

    dimension_columns = _process_dimension_columns(
        dimension_columns=dimension_columns, cube=cube
    )
    partition_by = _process_partition_by(
        partition_by=partition_by,
        cube=cube,
        all_available_columns=all_available_columns,
        indexed_columns=indexed_columns,
    )

    conditions_pre, conditions_post = _process_conditions(
        conditions=conditions,
        cube=cube,
        datasets=datasets,
        all_available_columns=all_available_columns,
        indexed_columns=indexed_columns,
    )

    payload_columns = _process_payload(
        payload_columns=payload_columns,
        all_available_columns=all_available_columns,
        cube=cube,
    )
    output_columns = tuple(
        sorted(
            set(partition_by)
            | set(dimension_columns)
            | set(payload_columns)
            | set(cube.partition_columns)
        )
    )

    return QueryIntention(
        dimension_columns=dimension_columns,
        partition_by=partition_by,
        conditions_pre=conditions_pre,
        conditions_post=conditions_post,
        output_columns=output_columns,
    )


@attr.s(frozen=True)
class QueryIntention:
    """
    Checked user intention during the query process.

    Parameters
    ----------
    dimension_columns: Tuple[str, ...]
        Real dimension columns.
    partition_by: Tuple[str, ...]
        Real partition-by columns, may be empty.
    conditions_pre: Dict[str, Conjuction]
        Conditions to be applied based on the index data alone.
    conditions_post: Dict[str, Conjuction]
        Conditions to be applied during the load process.
    output_columns: Tuple[str, ...]
        Output columns to be passed back to the user, in correct order.
    """

    dimension_columns = attr.ib(type=typing.Tuple[str, ...])
    partition_by = attr.ib(type=typing.Tuple[str, ...])
    conditions_pre = attr.ib(type=typing.Dict[str, Conjunction])
    conditions_post = attr.ib(type=typing.Dict[str, Conjunction])
    output_columns = attr.ib(type=typing.Tuple[str, ...])
