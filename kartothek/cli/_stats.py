import click

from kartothek.cli._utils import filter_items
from kartothek.cli._utils import to_bold as b
from kartothek.cli._utils import to_header as h
from kartothek.io.dask.bag_cube import collect_stats_bag

__all__ = ("stats",)


@click.option(
    "--include",
    help="Comma separated list of dataset-id to be scanned. e.g., ``--include enrich,enrich_cl`` "
    "also supports glob patterns",
    is_flag=False,
    metavar="<include>",
    type=click.STRING,
)
@click.option(
    "--exclude",
    help="Scan all datasets except items in this comma separated list. e.g., ``--exclude enrich,enrich_cl`` "
    "also supports glob patterns",
    is_flag=False,
    metavar="<exclude>",
    type=click.STRING,
)
@click.pass_context
def stats(ctx, include, exclude):
    """
    Collect technical statistic from cube.
    """
    cube = ctx.obj["cube"]
    store = ctx.obj["store"]
    all_datasets = set(ctx.obj["datasets"].keys())

    selected_datasets = filter_items("dataset", all_datasets, include, exclude)

    try:
        result = collect_stats_bag(
            cube=cube, store=store, datasets=selected_datasets
        ).compute()
    except RuntimeError as e:
        raise click.UsageError("Failed to collect stats: {e}".format(e=e))

    data = result[0]

    blobsize = 0
    files = 0

    for i, ktk_cube_dataset_id in enumerate(sorted(data.keys())):
        stats = data[ktk_cube_dataset_id]

        if i > 0:
            click.echo("")
        click.echo(h(ktk_cube_dataset_id))
        for what in sorted(stats.keys()):
            click.echo(b("{}:".format(what)) + "  {:,}".format(stats[what]))

        blobsize += stats["blobsize"]
        files += stats["files"]

    click.echo("")
    click.echo(h("__total__"))
    click.echo(b("blobsize:") + "  {:,}".format(blobsize))
    click.echo(b("files:") + "  {:,}".format(files))
