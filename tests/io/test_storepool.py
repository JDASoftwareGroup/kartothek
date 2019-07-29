import time
import weakref
from functools import partial

import pytest

from kartothek.io._storepool import MapStorepool, MapStorepoolThreaded


@pytest.mark.parametrize(
    "klass",
    [
        partial(MapStorepoolThreaded, nthreads=1),
        partial(MapStorepoolThreaded, nthreads=10),
        MapStorepool,
    ],
)
def test_storepool(klass):
    side_effects = []

    def f(x, store):
        side_effects.append(x)
        return x

    stores = []

    def store_factory():
        stores.append(None)

    num_jobs = 10
    with klass(store_factory=store_factory) as map_threaded:
        list(map_threaded(f, range(num_jobs)))
    assert len(stores) == 1
    assert len(side_effects) == num_jobs


@pytest.mark.parametrize("nthreads", range(1, 10))
def test_limit_concurrent_fetches(nthreads):
    side_effects = []

    def f(x, store):
        side_effects.append(x)
        return x

    num_jobs = 10
    with MapStorepoolThreaded(
        store_factory=lambda: None, nthreads=nthreads  # No need for a store here
    ) as map_threaded:
        res = map_threaded(f, range(num_jobs))
        next(res)
        sleep_time = 0.005
        # Give the function some time to execute. The side effects will be filled
        time.sleep(sleep_time)
        assert len(side_effects) == nthreads
        # Even when waiting for longer time no further function will be executed
        # A new function will only be called once a result is finished and a new one is fetched
        time.sleep(2 * sleep_time)
        assert len(side_effects) == nthreads
        next(res)
        time.sleep(sleep_time)
        assert len(side_effects) == nthreads + 1
        next(res)
        time.sleep(sleep_time)
        assert len(side_effects) == min(num_jobs, nthreads + 2)

        res = list(res)
        # In the end, all jobs are processed
        assert len(side_effects) == num_jobs


def test_propagate_exceptions():
    def f(x, store):
        if x == 5:
            raise RuntimeError("Intentional")
        elif x == 8:
            raise ValueError(
                "This should not happen because the tasks are killed before that"
            )

    nthreads = 2
    num_jobs = 10
    with pytest.raises(RuntimeError, match="Intentional"):
        with MapStorepoolThreaded(
            store_factory=lambda: None, nthreads=nthreads  # No need for a store here
        ) as map_threaded:
            list(map_threaded(f, range(num_jobs)))


@pytest.mark.parametrize("nthreads", range(1, 10))
def test_results_kept_after_usage(nthreads):
    links = weakref.WeakValueDictionary()

    class A:
        def __init__(self, val):
            self.val = val

    def f(x, store):
        # cannot weakref an integer
        obj = A(x)
        links[x] = obj
        return obj

    num_jobs = 10
    with MapStorepoolThreaded(
        store_factory=lambda: None, nthreads=nthreads  # No need for a store here
    ) as map_threaded:
        # need to trigger the generator
        sleep_time = 0.005
        for ix in map_threaded(f, range(num_jobs)):
            time.sleep(sleep_time)
            assert len(links) <= nthreads + 1


def test_threadpool_closes_on_exc():
    """
    Test that the threadpool and generator properly temrinate and don't block.
    This test should always be green but the test execution will deadlock during shutdown if there is an issue.
    """
    with MapStorepoolThreaded(
        store_factory=lambda: None, nthreads=10  # No need for a store here
    ) as map_threaded:
        try:
            for ix in map_threaded(lambda mp, store: None, range(10)):
                raise RuntimeError
        except Exception:
            pass
