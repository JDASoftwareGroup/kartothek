import logging
from typing import Callable, Generator, List, Union

import deprecation

from kartothek.core.dataset import DatasetMetadata
from kartothek.core.typing import StoreInput
from kartothek.core.utils import ensure_store
from kartothek.io_components.metapartition import MetaPartition
from kartothek.utils.migration_helpers import get_generic_function_deprecation_waring

LOGGER = logging.getLogger(__name__)

try:
    from typing_extensions import Literal  # type: ignore
except ImportError:
    from typing import Literal  # type: ignore


@deprecation.deprecated(
    deprecated_in="5.3",
    removed_in="6.0",
    details=get_generic_function_deprecation_waring(function_name="align_datasets"),
)
def align_datasets(
    left_dataset_uuid: str,
    right_dataset_uuid: str,
    store: StoreInput,
    match_how: Union[Literal["exact", "prefix", "all"], Callable] = "exact",
) -> Generator[List[MetaPartition], None, None]:
    """
    Determine dataset partition alignment

    Parameters
    ----------
    left_dataset_uuid
    right_dataset_uuid
    store
    match_how


    Yields
    ------
    List

    """
    store = ensure_store(store)
    left_dataset = DatasetMetadata.load_from_store(uuid=left_dataset_uuid, store=store)
    right_dataset = DatasetMetadata.load_from_store(
        uuid=right_dataset_uuid, store=store
    )

    metadata_version = left_dataset.metadata_version

    # Loop over the dataset with fewer partitions, treating its keys as
    # partition label prefixes
    if (
        callable(match_how)
        or match_how == "left"
        or (
            match_how == "prefix"
            and len(list(left_dataset.partitions.keys())[0])
            < len(list(right_dataset.partitions.keys())[0])
        )
    ):
        first_dataset = left_dataset
        second_dataset = right_dataset
    else:
        first_dataset = right_dataset
        second_dataset = left_dataset
    # The del statements are here to reduce confusion below
    del left_dataset
    del right_dataset

    # For every partition in the 'small' dataset, at least one partition match
    # needs to be found in the larger dataset.
    available_partitions = list(second_dataset.partitions.items())
    partition_stack = available_partitions[:]

    # TODO: write a test which protects against the following scenario!!
    # Sort the partition labels by length of the labels, starting with the
    # labels which are the longest. This way we prevent label matching for
    # similar partitions, e.g. cluster_100 and cluster_1. This, of course,
    # works only as long as the internal loop removes elements which were
    # matched already (here improperly called stack)
    for l_1 in sorted(first_dataset.partitions, key=len, reverse=True):
        p_1 = first_dataset.partitions[l_1]
        res = [
            MetaPartition.from_partition(
                partition=p_1, metadata_version=metadata_version
            )
        ]
        for parts in available_partitions:
            l_2, p_2 = parts
            if callable(match_how) and not match_how(l_1, l_2):
                continue
            if match_how == "exact" and l_1 != l_2:
                continue
            elif match_how == "prefix" and not l_2.startswith(l_1):
                LOGGER.debug("rejecting (%s, %s)", l_1, l_2)
                continue

            LOGGER.debug(
                "Found alignment between partitions " "(%s, %s) and" "(%s, %s)",
                first_dataset.uuid,
                p_1.label,
                second_dataset.uuid,
                p_2.label,
            )
            res.append(
                MetaPartition.from_partition(
                    partition=p_2, metadata_version=metadata_version
                )
            )

            # In exact or prefix matching schemes, it is expected to only
            # find one partition alignment. in this case reduce the size of
            # the inner loop
            if match_how in ["exact", "prefix"]:
                partition_stack.remove((l_2, p_2))
        # Need to copy, otherwise remove will alter the loop iterator
        available_partitions = partition_stack[:]
        if len(res) == 1:
            raise RuntimeError(
                "No matching partition for {} in dataset {} "
                "found".format(p_1, first_dataset)
            )
        yield res
