import click

from kartothek.cli._utils import filter_items
from kartothek.io.dask.bag_cube import delete_cube_bag

__all__ = ("delete",)


@click.option(
    "--include",
    help="Comma separated list of dataset-id to be deleted. e.g., ``--include enrich,enrich_cl`` "
    "also supports glob patterns",
    is_flag=False,
    metavar="<include>",
    type=click.STRING,
)
@click.option(
    "--exclude",
    help="Delete all datasets except items in this comma separated list. e.g., ``--exclude enrich,enrich_cl`` "
    "also supports glob patterns",
    is_flag=False,
    metavar="<exclude>",
    type=click.STRING,
)
@click.pass_context
def delete(ctx, include, exclude):
    """
    Delete cube from store.
    """
    cube = ctx.obj["cube"]
    store = ctx.obj["store"]
    all_datasets = set(ctx.obj["datasets"].keys())
    delete_datasets = filter_items("dataset", all_datasets, include, exclude)
    delete_cube_bag(cube=cube, store=store, datasets=delete_datasets).compute()
