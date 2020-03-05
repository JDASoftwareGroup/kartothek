import logging
import pathlib

from kartothek.io_components.utils import _instantiate_store

from .read import dispatch_metapartitions

LOGGER = logging.getLogger(__name__)


def align_datasets(dataset_uuids, store, predicates=None, match_how="exact"):
    """
    Determine dataset partition alignment

    Parameters
    ----------
    dataset_uuids : List[basestring]
    store : KeyValuestore or callable
    match_how : basestring or callable, {exact, prefix, all, callable}

    Yields
    ------
    list
    """
    store = _instantiate_store(store)

    # TODO: write a test which protects against the following scenario!!
    # Sort the partition labels by length of the labels, starting with the
    # labels which are the longest. This way we prevent label matching for
    # similar partitions, e.g. cluster_100 and cluster_1. This, of course,
    # works only as long as the internal loop removes elements which were
    # matched already (here improperly called stack)
    mp_0 = dispatch_metapartitions(dataset_uuids[0], store, predicates=predicates)

    for mp_0_i in mp_0:
        l_1 = mp_0_i.label
        res = [mp_0_i]
        for j in range(1, len(dataset_uuids)):
            mp_j = dispatch_metapartitions(
                dataset_uuids[j], store, predicates=predicates
            )
            for mp_j_i in mp_j:
                l_j = mp_j_i.label
                if callable(match_how) and not match_how(l_1, l_j):
                    continue
                if match_how == "prefix":
                    raise NotImplementedError
                elif match_how == "exact" and not (
                    pathlib.Path(l_1).parent == pathlib.Path(l_j).parent
                ):
                    continue

                LOGGER.debug(
                    "Found alignment between partitions " "(%s, %s) and" "(%s, %s)",
                    dataset_uuids[0],
                    l_1,
                    dataset_uuids[j],
                    l_j,
                )
                res.append(mp_j_i)
        yield res
