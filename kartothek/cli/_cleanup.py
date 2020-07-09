import click

from kartothek.io.dask.bag_cube import cleanup_cube_bag

__all__ = ("cleanup",)


@click.pass_context
def cleanup(ctx):
    """
    Remove non-required files from store.
    """
    cube = ctx.obj["cube"]
    store = ctx.obj["store"]

    cleanup_cube_bag(cube=cube, store=store).compute()
