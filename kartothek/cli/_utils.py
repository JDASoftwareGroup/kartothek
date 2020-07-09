import fnmatch
from functools import partial

import click
import storefact
import yaml

from kartothek.api.discover import discover_cube

__all__ = ("filter_items", "get_cube", "get_store", "to_bold", "to_header")


def get_cube(store, uuid_prefix):
    """
    Get cube from store.

    Parameters
    ----------
    uuid_prefix: str
        Dataset UUID prefix.
    store: Union[Callable[[], simplekv.KeyValueStore], simplekv.KeyValueStore]
        KV store.

    Returns
    -------
    cube: Cube
        Cube specification.
    datasets: Dict[str, kartothek.core.dataset.DatasetMetadata]
        All discovered datasets.

    Raises
    ------
    click.UsageError
        In case cube was not found.
    """
    try:
        return discover_cube(uuid_prefix, store)
    except ValueError as e:
        raise click.UsageError("Could not load cube: {e}".format(e=e))


def get_store(skv, store):
    """
    Get simplekv store from storefact config file.

    Parameters
    ----------
    skv: str
        Name of the storefact yaml. Normally ``'skv.yml'``.
    store: str
        ID of the store.

    Returns
    -------
    store_factory: Callable[[], simplekv.KeyValueStore]
        Store object.

    Raises
    ------
    click.UsageError
        In case something went wrong.
    """
    try:
        with open(skv, "rb") as fp:
            store_cfg = yaml.safe_load(fp)
    except IOError as e:
        raise click.UsageError("Could not open load store YAML: {e}".format(e=e))
    except yaml.YAMLError as e:
        raise click.UsageError("Could not parse provided YAML file: {e}".format(e=e))

    if store not in store_cfg:
        raise click.UsageError(
            "Could not find store {store} in {skv}".format(store=store, skv=skv)
        )

    return partial(storefact.get_store, **store_cfg[store])


def _match_pattern(what, items, pattern):
    """
    Match given pattern against given items.

    Parameters
    ----------
    what: str
        Describes what is filterd.
    items: Iterable[str]
        Items to be filtered
    include_pattern: str
        Comma separated items which should be included. Can contain glob patterns.
    """
    result = set()
    for part in pattern.split(","):
        found = set(fnmatch.filter(items, part.strip()))
        if not found:
            raise click.UsageError(
                "Could not find {what} {part}".format(what=what, part=part)
            )
        result |= found
    return result


def filter_items(what, items, include_pattern=None, exclude_pattern=None):
    """
    Filter given string items based on include and exclude patterns

    Parameters
    ----------
    what: str
        Describes what is filterd.
    items: Iterable[str]
        Items to be filtered
    include_pattern: str
        Comma separated items which should be included. Can contain glob patterns.
    exclude_pattern: str
        Comma separated items which should be excluded. Can contain glob patterns.

    Returns
    -------
    filtered_datasets: Set[str]
        Filtered set of items after applying include and exclude patterns
    """
    items = set(items)

    if include_pattern is not None:
        include_datasets = _match_pattern(what, items, include_pattern)
    else:
        include_datasets = items

    if exclude_pattern is not None:
        exclude_datasets = _match_pattern(what, items, exclude_pattern)
    else:
        exclude_datasets = set()

    return include_datasets - exclude_datasets


def to_header(s):
    """
    Create header.

    Parameters
    ----------
    s: str
        Header content.

    Returns
    -------
    s: str
        Header content including terminal escpae sequences.
    """
    return click.style(s, bold=True, underline=True, fg="yellow")


def to_bold(s):
    """
    Create bold text.

    Parameters
    ----------
    s: str
        Bold text content.

    Returns
    -------
    s: str
        Given text including terminal escpae sequences.
    """
    return click.style(s, bold=True)
