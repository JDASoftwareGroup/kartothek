from typing import Dict, Iterable

from kartothek.core.dataset import DatasetMetadata

__all__ = ("check_existing_datasets",)


def check_existing_datasets(
    existing_datasets: Dict[str, DatasetMetadata], ktk_cube_dataset_ids: Iterable[str]
):
    """
    Check existing datasets for append operation to ensure they all exist.

    Parameters
    ----------
    existing_datasets:
        Existing datasets.
    ktk_cube_dataset_ids:
        Datasets that user wants to append data to.

    Raises
    ------
    ValueError: In case a dataset does not exist yet.
    """
    unknown_datasets = set(ktk_cube_dataset_ids) - set(existing_datasets.keys())
    if unknown_datasets:
        raise ValueError(
            "Unknown / non-existing datasets: {}".format(
                ", ".join(sorted(unknown_datasets))
            )
        )
