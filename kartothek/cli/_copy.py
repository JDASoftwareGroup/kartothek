import click

from kartothek.cli._utils import filter_items, get_cube, get_store
from kartothek.io.dask.bag_cube import copy_cube_bag, delete_cube_bag

__all__ = ("copy",)


@click.option("--tgt_store", required=True, help="Target store to use.")
@click.option(
    "--overwrite/--no-overwrite",
    default=False,
    help=(
        "Flags if potentially present cubes in ``tgt_store`` are overwritten. If ``--no-overwrite`` is given (default) "
        "and a cube is already present, the operation will fail."
    ),
    show_default=True,
)
@click.option(
    "--cleanup/--no-cleanup",
    default=True,
    help=(
        "Flags if in case of an overwrite operation, the cube in ``tgt_store`` will first be removed so no previously "
        "tracked files will be present after the copy operation."
    ),
    show_default=True,
)
@click.option(
    "--include",
    help="Comma separated list of dataset-id to be copied. e.g., ``--include enrich,enrich_cl`` "
    "also supports glob patterns",
    is_flag=False,
    metavar="<include>",
    type=click.STRING,
)
@click.option(
    "--exclude",
    help="Copy all datasets except items in this comma separated list. e.g., ``--exclude enrich,enrich_cl`` "
    "also supports glob patterns",
    is_flag=False,
    metavar="<exclude>",
    type=click.STRING,
)
@click.pass_context
def copy(ctx, tgt_store, overwrite, cleanup, include, exclude):
    """
    Copy cube from one store to another.
    """
    cube = ctx.obj["cube"]
    skv = ctx.obj["skv"]
    store = ctx.obj["store"]
    store_name = ctx.obj["store_name"]
    all_datasets = set(ctx.obj["datasets"].keys())

    if store_name == tgt_store:
        raise click.UsageError("Source and target store must be different.")

    tgt_store = get_store(skv, tgt_store)
    selected_datasets = filter_items("dataset", all_datasets, include, exclude)

    if overwrite:
        try:
            cube2, _ = get_cube(tgt_store, cube.uuid_prefix)
            if cleanup:
                click.secho(
                    "Deleting existing datasets {selected_datasets} from target store...".format(
                        selected_datasets=",".join(selected_datasets)
                    )
                )
                delete_cube_bag(
                    cube=cube2, store=tgt_store, datasets=selected_datasets
                ).compute()
            else:
                click.secho("Skip cleanup, leave old existing cube.")
        except click.UsageError:
            # cube not found, nothing to cleanup
            pass

    click.secho(
        "Copy datasets {selected_datasets} to target store...".format(
            selected_datasets=",".join(selected_datasets)
        )
    )
    try:
        copy_cube_bag(
            cube=cube,
            src_store=store,
            tgt_store=tgt_store,
            overwrite=overwrite,
            datasets=selected_datasets,
        ).compute()
    except (RuntimeError, ValueError) as e:
        raise click.UsageError("Failed to copy cube: {e}".format(e=e))
