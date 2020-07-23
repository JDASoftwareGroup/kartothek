"""
Kartothek CLI code.

.. important::
    This module does not contain any public APIs.

Kartothek comes with a CLI tool named ``kartothek_cube``. To use it, create an YAML file that contains a dictionary of `storefact`_
stores (keys are names of the store and the values are dicts that contain the store config). ``Kartothek`` uses a `YAML`_
file called ``skv.yml`` and a store called ``dataset`` by default, but you may pass ``--skv`` and ``--store`` to change
these. An example file could look like:

.. code-block:: yaml

   dataset:
      type: hazure
      account_name: my_account_name
      account_key: foobar
      container: my_container
      use_sas: False
      create_if_missing: False

The CLI uses `Dask`_ to parallelize some operations and defaults to the number of CPU cores. You can control the number
of threads using ``-j``.

In the following section you find a list description of all ``kartothek_cube`` operations.

.. click:: kartothek.cli:cli
   :prog: kartothek_cube
   :show-nested:


.. _Dask: https://docs.dask.org/
.. _storefact: https://github.com/blue-yonder/storefact
.. _YAML: https://yaml.org/
"""
import logging
from multiprocessing.pool import ThreadPool

import click
import dask
import pandas as pd
from dask.diagnostics import ProgressBar

from kartothek.cli._cleanup import cleanup
from kartothek.cli._copy import copy
from kartothek.cli._delete import delete
from kartothek.cli._index import index
from kartothek.cli._info import info
from kartothek.cli._query import query
from kartothek.cli._stats import stats
from kartothek.cli._utils import get_cube, get_store

__all__ = ("cli",)


@click.group(context_settings=dict(help_option_names=["-h", "--help"]))
@click.option(
    "--skv", default="skv.yml", help="Storefact config file.", show_default=True
)
@click.option("--store", default="dataset", help="Store to use.", show_default=True)
@click.option(
    "--n_threads",
    "-j",
    default=0,
    type=int,
    help="Number of threads to use (use 0 for number of cores).",
    show_default=True,
)
@click.option(
    "--color",
    type=click.Choice(["always", "auto", "off"]),
    default="auto",
    help="Whether to use colorized outputs or not. Use ``always``, ``auto`` (default), or ``off``.",
    show_default=True,
)
@click.argument("cube")
@click.pass_context
def cli(ctx, store, cube, skv, n_threads, color):
    """
    Execute certain operations on the given Kartothek cube.

    If possible, the operations will be performed in parallel on the current machine.
    """
    ctx.ensure_object(dict)

    store_obj = get_store(skv, store)
    cube, datasets = get_cube(store_obj, cube)

    dask.config.set(scheduler="threads")
    if n_threads > 0:
        dask.config.set(pool=ThreadPool(n_threads))

    if color == "always":
        ctx.color = True
    elif color == "off":
        ctx.color = False

    pbar = ProgressBar()
    pbar.register()
    ctx.call_on_close(pbar.unregister)

    # silence extremely verbose azure logging
    azure_logger = logging.getLogger("azure.storage.common.storageclient")
    azure_logger.setLevel(logging.FATAL)

    # pandas perf tuning
    chained_assignment_old = pd.options.mode.chained_assignment

    def reset_pd():
        pd.options.mode.chained_assignment = chained_assignment_old

    ctx.call_on_close(reset_pd)
    pd.options.mode.chained_assignment = None

    ctx.obj["skv"] = skv
    ctx.obj["store"] = store_obj
    ctx.obj["store_name"] = store
    ctx.obj["cube"] = cube
    ctx.obj["datasets"] = datasets
    ctx.obj["pbar"] = pbar


cli.command()(cleanup)
cli.command()(copy)
cli.command()(delete)
cli.command()(index)
cli.command()(info)
cli.command()(query)
cli.command()(stats)


if __name__ == "__main__":
    cli()
