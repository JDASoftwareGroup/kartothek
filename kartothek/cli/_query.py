from functools import partial

import click
import numpy as np  # noqa
import pandas as pd
from prompt_toolkit import prompt
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.validation import ValidationError, Validator

from kartothek.core.cube.conditions import Conjunction
from kartothek.io.dask.bag_cube import query_cube_bag
from kartothek.io_components.metapartition import SINGLE_TABLE
from kartothek.utils.ktk_adapters import get_dataset_columns

__all__ = ("query",)


_history_conditions = InMemoryHistory()
_history_payload = InMemoryHistory()


@click.pass_context
def query(ctx):
    """
    Interactive cube queries into IPython.
    """
    cube = ctx.obj["cube"]
    datasets = ctx.obj["datasets"]
    store = ctx.obj["store"]

    store_instance = store()

    datasets = {
        ktk_cube_dataset_id: ds.load_all_indices(store_instance)
        for ktk_cube_dataset_id, ds in datasets.items()
    }

    all_columns = set()
    all_types = {}
    for ds in datasets.values():
        cols = get_dataset_columns(ds)
        all_columns |= cols
        for col in cols:
            all_types[col] = ds.table_meta[SINGLE_TABLE].field(col).type

    ipython = _get_ipython()

    conditions = None
    payload_columns = []

    while True:
        conditions = _ask_conditions(conditions, all_columns, all_types)
        payload_columns = _ask_payload(payload_columns, all_columns)

        result = query_cube_bag(
            cube=cube,
            store=store,
            conditions=conditions,
            datasets=datasets,
            payload_columns=payload_columns,
        ).compute()

        if not result:
            click.secho("No data found.", bold=True, fg="red")
            continue

        df = result[0]
        _shell(df, ipython)


def _get_ipython():
    try:
        import IPython  # noqa

        return IPython
    except Exception as e:
        raise click.UsageError("Could not load IPython: {e}".format(e=e))


def _shell(df, ipython):
    pd.set_option("display.width", None)
    pd.set_option("display.max_rows", 0)

    try:
        ipython.embed(banner1="Shell:")
    except ipython.terminal.embed.KillEmbeded:
        raise KeyboardInterrupt()


class _ValidatorFromParse(Validator):
    def __init__(self, f_parse):
        super(_ValidatorFromParse, self).__init__()
        self.f_parse = f_parse

    def validate(self, document):
        txt = document.text
        try:
            self.f_parse(txt)
        except ValueError as e:
            raise ValidationError(message=str(e))


def _ask_payload(payload_columns, all_columns):
    def _parse(txt):
        if txt == "__all__":
            return sorted(all_columns)

        cleaned = {s.strip() for s in txt.split(",")}
        cols = {s for s in cleaned if s}
        missing = cols - set(all_columns)
        if missing:
            raise ValueError(
                "Unknown: {missing}".format(missing=", ".join(sorted(missing)))
            )
        return sorted(cols)

    if set(payload_columns) == all_columns:
        default = "__all__"
    else:
        default = ",".join(payload_columns)

    txt = prompt(
        message="Payload Columns: ",
        history=_history_payload,
        default=default,
        completer=WordCompleter(sorted(all_columns) + ["__all__"]),
        validator=_ValidatorFromParse(_parse),
    )
    return _parse(txt)


def _ask_conditions(conditions, all_columns, all_types):
    txt = prompt(
        message="Conditions: ",
        history=_history_conditions,
        default=str(conditions) if conditions is not None else "",
        completer=WordCompleter(sorted(all_columns)),
        validator=_ValidatorFromParse(
            partial(Conjunction.from_string, all_types=all_types)
        ),
    )
    return Conjunction.from_string(txt, all_types)
