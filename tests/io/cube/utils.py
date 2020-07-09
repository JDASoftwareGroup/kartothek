import pickle
from collections import defaultdict

import dask
import dask.bag as db
import dask.dataframe as ddf


def pickle_roundtrip(obj):
    s = pickle.dumps(obj, pickle.HIGHEST_PROTOCOL)
    return pickle.loads(s)


def _return_obj(obj):
    return obj


def wrap_bag_write(func, blocksize):
    blocksize_outer = blocksize

    def _driver(data, no_run=False, blocksize=None, *args, **kwargs):
        if blocksize is None:
            blocksize = blocksize_outer

        if not isinstance(data, list):
            data = [data]
        _add_dataset_ids(data, kwargs)

        data = db.from_sequence(data, partition_size=blocksize)

        b = func(data, *args, **kwargs)
        if no_run:
            return
        b = pickle_roundtrip(b)

        result = b.compute()
        assert len(result) == 1
        return result[0]

    return _driver


def wrap_bag_read(f, blocksize):
    blocksize_outer = blocksize

    def _driver(no_run=False, blocksize=None, **kwargs):
        if blocksize is None:
            blocksize = blocksize_outer

        b = f(blocksize=blocksize, **kwargs)
        if no_run:
            return
        b = pickle_roundtrip(b)

        result = b.compute()
        if not isinstance(result, list):
            return [result]
        else:
            return result

    return _driver


def wrap_bag_stats(f, blocksize):
    blocksize_outer = blocksize

    def _driver(no_run=False, blocksize=None, **kwargs):
        if blocksize is None:
            blocksize = blocksize_outer

        b = f(blocksize=blocksize, **kwargs)
        if no_run:
            return
        b = pickle_roundtrip(b)

        result = b.compute()
        return result[0]

    return _driver


def wrap_ddf_write(f):
    def _driver(data, cube, no_run=False, **kwargs):
        if not isinstance(data, list):
            data = [data]
        dfs = defaultdict(list)
        for part in data:
            if not isinstance(part, dict):
                part = {cube.seed_dataset: part}
            for ktk_cube_dataset_id, df in part.items():
                dfs[ktk_cube_dataset_id].append(dask.delayed(_return_obj)(df))

        ddfs = {
            ktk_cube_dataset_id: ddf.from_delayed(delayed_objs)
            for ktk_cube_dataset_id, delayed_objs in dfs.items()
        }

        delayed = f(data=ddfs, cube=cube, **kwargs)
        if no_run:
            return
        delayed = pickle_roundtrip(delayed)

        return delayed.compute()

    return _driver


def wrap_ddf_read(f):
    def _driver(partition_by=None, no_run=False, **kwargs):
        ddf = f(partition_by=partition_by, **kwargs)
        if no_run:
            return
        ddf = pickle_roundtrip(ddf)

        dfs = [d.compute() for d in ddf.to_delayed()]
        return [df for df in dfs if not df.empty]

    return _driver


def _add_dataset_ids(data, kwargs):
    if "ktk_cube_dataset_ids" not in kwargs:
        ktk_cube_dataset_ids = set()
        for part in data:
            if isinstance(part, dict):
                ktk_cube_dataset_ids |= set(part.keys())

        if ktk_cube_dataset_ids:
            kwargs["ktk_cube_dataset_ids"] = sorted(ktk_cube_dataset_ids)


def wrap_bag_copy(f, blocksize):
    blocksize_outer = blocksize

    def _driver(no_run=False, blocksize=None, **kwargs):
        if blocksize is None:
            blocksize = blocksize_outer

        b = f(blocksize=blocksize, **kwargs)
        if no_run:
            return
        b = pickle_roundtrip(b)

        b.compute()

    return _driver


def wrap_bag_delete(f, blocksize):
    blocksize_outer = blocksize

    def _driver(no_run=False, blocksize=None, **kwargs):
        if blocksize is None:
            blocksize = blocksize_outer

        b = f(blocksize=blocksize, **kwargs)
        if no_run:
            return
        b = pickle_roundtrip(b)

        b.compute()

    return _driver
