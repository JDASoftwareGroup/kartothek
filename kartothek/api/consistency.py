"""
Methods to check preserved cube for consistency.
"""
from collections import defaultdict
from copy import copy
from functools import reduce

from kartothek.core.common_metadata import validate_shared_columns
from kartothek.core.cube.constants import KTK_CUBE_METADATA_VERSION
from kartothek.core.index import ExplicitSecondaryIndex, PartitionIndex
from kartothek.io_components.metapartition import SINGLE_TABLE
from kartothek.utils.ktk_adapters import get_dataset_columns

__all__ = ("check_datasets", "get_cube_payload", "get_payload_subset")


def _check_datasets(datasets, f, expected, what):
    """
    Check datasets with given function and raise ``ValueError`` in case of an issue.

    Parameters
    ----------
    datasets: Dict[str, kartothek.core.dataset.DatasetMetadata]
        Datasets.
    f: Callable[[kartothek.core.dataset.DatasetMetadata], Any]
        Transformer for dataset.
    expected: Any
        Value that is expected to be returned by ``f``.
    what: str
        Description of what is currently checked.

    Raises
    ------
    ValueError: In case any issue was found.
    """
    no = [name for name, ds in datasets.items() if f(ds) != expected]
    if no:

        def _fmt(obj):
            if isinstance(obj, set):
                return ", ".join(sorted(obj))
            elif isinstance(obj, (list, tuple)):
                return ", ".join(obj)
            else:
                return str(obj)

        raise ValueError(
            "Invalid datasets because {what} is wrong. Expected {expected}: {datasets}".format(
                what=what,
                expected=_fmt(expected),
                datasets=", ".join(
                    "{name} ({actual})".format(
                        name=name, actual=_fmt(f(datasets[name]))
                    )
                    for name in sorted(no)
                ),
            )
        )


def _check_overlap(datasets, cube):
    """
    Check that datasets have not overlapping payload columns.

    Parameters
    ----------
    datasets: Dict[str, kartothek.core.dataset.DatasetMetadata]
        Datasets.
    cube: kartothek.core.cube.cube.Cube
        Cube specification.

    Raises
    ------
    ValueError: In case of overlapping payload columns.
    """
    payload_columns = defaultdict(list)
    for ktk_cube_dataset_id, ds in datasets.items():
        for col in get_payload_subset(get_dataset_columns(ds), cube):
            payload_columns[col].append(ktk_cube_dataset_id)
    payload_columns = {
        col: ktk_cube_dataset_ids
        for col, ktk_cube_dataset_ids in payload_columns.items()
        if len(ktk_cube_dataset_ids) > 1
    }
    if payload_columns:
        raise ValueError(
            "Found columns present in multiple datasets:{}".format(
                "\n".join(
                    " - {col}: {ktk_cube_dataset_ids}".format(
                        col=col,
                        ktk_cube_dataset_ids=", ".join(sorted(payload_columns[col])),
                    )
                    for col in sorted(payload_columns.keys())
                )
            )
        )


def _check_dimension_columns(datasets, cube):
    """
    Check if required dimension are present in given datasets.

    For the seed dataset all dimension columns must be present. For all other datasets at least 1 dimension column must
    be present.

    Parameters
    ----------
    datasets: Dict[str, kartothek.core.dataset.DatasetMetadata]
        Datasets.
    cube: kartothek.core.cube.cube.Cube
        Cube specification.

    Raises
    ------
    ValueError: In case dimension columns are broken.
    """
    for ktk_cube_dataset_id in sorted(datasets.keys()):
        ds = datasets[ktk_cube_dataset_id]
        columns = get_dataset_columns(ds)
        if ktk_cube_dataset_id == cube.seed_dataset:
            missing = set(cube.dimension_columns) - columns
            if missing:
                raise ValueError(
                    'Seed dataset "{ktk_cube_dataset_id}" has missing dimension columns: {missing}'.format(
                        ktk_cube_dataset_id=ktk_cube_dataset_id,
                        missing=", ".join(sorted(missing)),
                    )
                )
        else:
            present = set(cube.dimension_columns) & columns
            if len(present) == 0:
                raise ValueError(
                    (
                        'Dataset "{ktk_cube_dataset_id}" must have at least 1 of the following dimension columns: '
                        "{dims}"
                    ).format(
                        ktk_cube_dataset_id=ktk_cube_dataset_id,
                        dims=", ".join(cube.dimension_columns),
                    )
                )


def _check_partition_columns(datasets, cube):
    """
    Check if required partitions columns are present in given datasets.

    For the seed dataset all partition columns must be present. For all other datasets at least 1 partition column must
    be present.

    Parameters
    ----------
    datasets: Dict[str, kartothek.core.dataset.DatasetMetadata]
        Datasets.
    cube: kartothek.core.cube.cube.Cube
        Cube specification.

    Raises
    ------
    ValueError: In case partition columns are broken.
    """
    for ktk_cube_dataset_id in sorted(datasets.keys()):
        ds = datasets[ktk_cube_dataset_id]
        columns = set(ds.partition_keys)

        if ktk_cube_dataset_id == cube.seed_dataset:
            missing = set(cube.partition_columns) - columns
            if missing:
                raise ValueError(
                    'Seed dataset "{ktk_cube_dataset_id}" has missing partition columns: {missing}'.format(
                        ktk_cube_dataset_id=ktk_cube_dataset_id,
                        missing=", ".join(sorted(missing)),
                    )
                )

        unspecified_partition_columns = (
            get_dataset_columns(ds) - set(ds.partition_keys)
        ) & set(cube.partition_columns)
        if unspecified_partition_columns:
            raise ValueError(
                f"Unspecified but provided partition columns in {ktk_cube_dataset_id}: "
                f"{', '.join(sorted(unspecified_partition_columns))}"
            )


def _check_indices(datasets, cube):
    """
    Check if required indices are present in given datasets.

    For all datasets the primary indices must be equal to ``ds.partition_keys``. For the seed dataset secondary
    indices for all dimension columns are expected.

    Parameters
    ----------
    datasets: Dict[str, kartothek.core.dataset.DatasetMetadata]
        Datasets.
    cube: kartothek.core.cube.cube.Cube
        Cube specification.

    Raises
    ------
    ValueError: In case indices are broken.
    """
    for ktk_cube_dataset_id in sorted(datasets.keys()):
        ds = datasets[ktk_cube_dataset_id]
        primary_indices = ds.partition_keys
        columns = get_dataset_columns(ds)
        secondary_indices = set()
        any_indices = set(cube.index_columns) & columns

        if ktk_cube_dataset_id == cube.seed_dataset:
            secondary_indices |= set(cube.dimension_columns)

        for types, elements in (
            ((PartitionIndex,), primary_indices),
            ((ExplicitSecondaryIndex,), secondary_indices),
            ((ExplicitSecondaryIndex, PartitionIndex), any_indices),
        ):
            tname = " or ".join(t.__name__ for t in types)

            # it seems that partition indices are not always present (e.g. for empty datasets), so add partition keys to
            # the set
            indices = copy(ds.indices)
            if PartitionIndex in types:
                for pk in ds.partition_keys:
                    if pk not in indices:
                        indices[pk] = "dummy"

            for e in sorted(elements):
                if e not in indices:
                    raise ValueError(
                        '{tname} "{e}" is missing in dataset "{ktk_cube_dataset_id}".'.format(
                            tname=tname, e=e, ktk_cube_dataset_id=ktk_cube_dataset_id
                        )
                    )

                idx = indices[e]
                t2 = type(idx)
                tname2 = t2.__name__
                if (idx != "dummy") and (not isinstance(idx, types)):
                    raise ValueError(
                        '"{e}" in dataset "{ktk_cube_dataset_id}" is of type {tname2} but should be {tname}.'.format(
                            tname=tname,
                            tname2=tname2,
                            e=e,
                            ktk_cube_dataset_id=ktk_cube_dataset_id,
                        )
                    )


def check_datasets(datasets, cube):
    """
    Apply sanity checks to persisteted Karothek datasets.

    The following checks will be applied:

    - seed dataset present
    - metadata version correct
    - only the cube-specific table is present
    - partition keys are correct
    - no overlapping payload columns exists
    - datatypes are consistent
    - dimension columns are present everywhere
    - required index structures are present (more are allowed)

      - ``PartitionIndex`` for every partition key
      - for seed dataset, ``ExplicitSecondaryIndex`` for every dimension column
      - for all datasets, ``ExplicitSecondaryIndex`` for every index column

    Parameters
    ----------
    datasets: Dict[str, kartothek.core.dataset.DatasetMetadata]
        Datasets.
    cube: kartothek.core.cube.cube.Cube
        Cube specification.
    existing_seed_dataset: Optional[kartothek.core.dataset.DatasetMetadata]
        Optional existing seed dataset metadata.

    Returns
    -------
    datasets: Dict[str, kartothek.core.dataset.DatasetMetadata]
        Same as input, but w/ partition indices loaded.

    Raises
    ------
    ValueError
        If sanity check failed.
    """
    if cube.seed_dataset not in datasets:
        raise ValueError('Seed data ("{}") is missing.'.format(cube.seed_dataset))

    _check_datasets(
        datasets=datasets,
        f=lambda ds: ds.metadata_version,
        expected=KTK_CUBE_METADATA_VERSION,
        what="metadata version",
    )
    datasets = {name: ds.load_partition_indices() for name, ds in datasets.items()}
    _check_datasets(
        datasets=datasets,
        f=lambda ds: set(ds.table_meta.keys()),
        expected={SINGLE_TABLE},
        what="table",
    )
    _check_overlap(datasets, cube)

    # check column types
    validate_shared_columns([ds.table_meta[SINGLE_TABLE] for ds in datasets.values()])

    _check_partition_columns(datasets, cube)
    _check_dimension_columns(datasets, cube)
    _check_indices(datasets, cube)

    return datasets


def get_payload_subset(columns, cube):
    """
    Get payload column subset from a given set of columns.

    Parameters
    ----------
    columns: Iteratable[str]
        Columns.
    cube: kartothek.core.cube.cube.Cube
        Cube specification.

    Returns
    -------
    payload: Set[str]
        Payload columns.
    """
    return set(columns) - set(cube.dimension_columns) - set(cube.partition_columns)


def get_cube_payload(datasets, cube):
    """
    Get payload columns of the whole cube.

    Parameters
    ----------
    datasets: Dict[str, kartothek.core.dataset.DatasetMetadata]
        Datasets.
    cube: kartothek.core.cube.cube.Cube
        Cube specification.

    Returns
    -------
    payload: Set[str]
        Payload columns.
    """
    return reduce(
        set.union,
        (get_payload_subset(get_dataset_columns(ds), cube) for ds in datasets.values()),
        set(),
    )
