import json

import click

from kartothek.cli._utils import to_bold as b
from kartothek.cli._utils import to_header as h
from kartothek.io_components.metapartition import SINGLE_TABLE
from kartothek.utils.ktk_adapters import get_dataset_columns

__all__ = ("info",)


@click.pass_context
def info(ctx):
    """
    Show certain infos about the cube.
    """
    cube = ctx.obj["cube"]
    datasets = ctx.obj["datasets"]

    seed_ds = datasets[cube.seed_dataset]
    seed_schema = seed_ds.table_meta[SINGLE_TABLE]

    click.echo(h("Infos"))
    click.echo(b("UUID Prefix:") + "        {}".format(cube.uuid_prefix))
    click.echo(
        b("Dimension Columns:") + _collist_string(cube.dimension_columns, seed_schema)
    )
    click.echo(
        b("Partition Columns:") + _collist_string(cube.partition_columns, seed_schema)
    )
    click.echo(b("Index Columns:") + _collist_string_index(cube, datasets))
    click.echo(b("Seed Dataset:") + "      {}".format(cube.seed_dataset))

    for ktk_cube_dataset_id in sorted(datasets.keys()):
        _info_dataset(ktk_cube_dataset_id, datasets[ktk_cube_dataset_id], cube)


def _info_dataset(ktk_cube_dataset_id, ds, cube):
    click.echo("")
    click.echo(h("Dataset: {}".format(ktk_cube_dataset_id)))

    ds = ds.load_partition_indices()
    schema = ds.table_meta[SINGLE_TABLE]
    all_cols = get_dataset_columns(ds)
    payload_cols = sorted(
        all_cols - (set(cube.dimension_columns) | set(cube.partition_columns))
    )
    dim_cols = sorted(set(cube.dimension_columns) & all_cols)

    click.echo(b("Partition Keys:") + _collist_string(ds.partition_keys, schema))

    click.echo(b("Partitions:") + " {}".format(len(ds.partitions)))

    click.echo(
        b("Metadata:")
        + "\n{}".format(
            "\n".join(
                "  {}".format(line)
                for line in json.dumps(
                    ds.metadata, indent=2, sort_keys=True, separators=(",", ": ")
                ).split("\n")
            )
        )
    )

    click.echo(b("Dimension Columns:") + _collist_string(dim_cols, schema))

    click.echo(b("Payload Columns:") + _collist_string(payload_cols, schema))


def _collist_string(cols, schema):
    if cols:
        return "\n" + "\n".join(
            "  - {c}: {t}".format(c=c, t=schema.field_by_name(c).type) for c in cols
        )
    else:
        return ""


def _collist_string_index(cube, datasets):
    lines = []
    for col in sorted(cube.index_columns):
        for ktk_cube_dataset_id in sorted(datasets.keys()):
            ds = datasets[ktk_cube_dataset_id]
            schema = ds.table_meta[SINGLE_TABLE]
            if col in schema.names:
                lines.append(
                    "  - {c}: {t}".format(c=col, t=schema.field_by_name(col).type)
                )
                break
    return "\n" + "\n".join(lines)
