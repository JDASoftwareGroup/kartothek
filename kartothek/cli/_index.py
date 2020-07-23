import click

from kartothek.cli._utils import filter_items
from kartothek.io.dask.bag import build_dataset_indices__bag
from kartothek.utils.ktk_adapters import get_dataset_columns

__all__ = ("index",)


@click.argument("dataset")
@click.argument("columns")
@click.pass_context
def index(ctx, dataset, columns):
    """
    Build index for given columns.
    """
    store = ctx.obj["store"]
    datasets = ctx.obj["datasets"]

    if dataset not in datasets:
        raise click.UsageError(
            "Could not find dataset {dataset}, known datasets are {known}".format(
                dataset=dataset, known=", ".join(sorted(datasets))
            )
        )

    ds = datasets[dataset]
    columns = filter_items("column", get_dataset_columns(ds), columns, None)
    build_dataset_indices__bag(
        dataset_uuid=ds.uuid, store=store, columns=columns
    ).compute()
