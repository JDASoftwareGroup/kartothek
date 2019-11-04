from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from queue import Queue
from threading import Condition


class MapStorepoolThreaded:
    def __init__(self, nthreads, store_factory):
        """
        Context manager providing a map function which parallelizes the tasks
        using threads. The implementation guarantees that at most nthreads partitions are held in memory.
        If the pool is saturated, all threads will sleep until a results is done processing and control
        flow is returned tothe generator.

        To avoid the opening a connection to a store for every thread
        individually, initialized stores are pooled and the threads can re-use
        stores/connection.
        The initialized store object don't need to be thread safe.

        Usage::

            with MapStorepoolThreaded(5, store_factory) as map_threaded:
                mps = map_threaded(
                    MetaPartition.load_dataframes,
                    mps,
                    # pass arguments for load_dataframes
                    columns=columns,
                    ...
                )
                for mp in mps:
                    # If pool is saturated, no more partitions will be fetched
                    yield mp
                    # after yield returns control one thread will awake and fetch a new partition

        note::

            This class is for internal use only, doesn't offer a stable API and
            may be removed or changed without further warning!

        Parameters
        ----------
        nthreads: int
            The number of threads to be spawned and maximum number of results in memory.
        store_factory: callable
            A callable creating a simpleKV store object
        """
        self.nthreads = nthreads
        self.thread_pool = None
        self._futures = []
        self.cond = Condition()
        self.result_queue = Queue()
        self.store_factory = store_factory
        self.store_pool = deque()
        self._abort = False

    def __enter__(self):
        self.thread_pool = ThreadPoolExecutor(self.nthreads)
        return self

    def __call__(self, func, iterable, *args, **kwargs):
        self._futures = [
            self.thread_pool.submit(
                self._process_job_with_storepool, func, mp, *args, **kwargs
            )
            for mp in iterable
        ]
        with self.cond:
            # This controls how many results are allowed to be concurrently
            # fetched
            self.cond.notify(self.nthreads)
        return iter(self)

    def __exit__(self, exc_type, exc_val, tb):
        # If the ctx is left ungracefully we need to clean up the scheduled tasks since otherwise the threadpool
        # cannot properly close and will block. The logic will cancel all scheduled futures and sets the abort flag.
        # All sleeping threads are then woken and are allowed to gracefully finish before the ThreadPool shuts down.
        for fut in self._futures:
            fut.cancel()
        with self.cond:
            self._abort = False
            self.cond.notify_all()
        self.thread_pool.shutdown()

    def __iter__(self):
        for fut in as_completed(self._futures):
            exc = fut.exception()
            if exc:
                raise exc
            yield self.result_queue.get()
            with self.cond:
                self.cond.notify()

    def _process_job_with_storepool(self, method, mp, *args, **kwargs):
        with self.cond:
            self.cond.wait()
            # If an exception is raised while this thread was asleep and the
            # pool is shut down, don't execute any logic
            if self._abort:
                return
            # The check for available stores in the pool
            # needs to be within the lock context
            if len(self.store_pool):
                store = self.store_pool.pop()
            else:
                store = self.store_factory()

        res = method(mp, store=store, *args, **kwargs)
        self.store_pool.append(store)
        self.result_queue.put(res)


class MapStorepool:
    def __init__(self, store_factory):
        """
        A contextmanager which offers a specialized map function which pool initialized stores/connections and allows
        them to be reused for every item in the map.

        Usage::

            with MapStorepool(store_factory) as _map:
                mps = _map(
                    MetaPartition.load_dataframes,
                    mps,
                    # pass arguments for load_dataframes
                    columns=columns,
                    ...
                )
                for mp in mps:
                    # Do somethint with the result. A new thread will only start running once the yield returns control to the generator
                    yield mp

        note::

            This class is for internal use only, doesn't offer a stable API and
            may be removed or changed without further warning!

        Parameters
        ----------
        store_factory: callable
            Callable to create a simplekv store
        """
        self.store = store_factory()
        self._results = []

    def __enter__(self):
        return self

    def __call__(self, func, iterable, *args, **kwargs):
        self._results = map(
            partial(self._process_job_with_storepool, func, *args, **kwargs), iterable
        )
        return iter(self)

    def __exit__(self, exc_type, exc_val, tb):
        pass

    def __iter__(self):
        return iter(self._results)

    def _process_job_with_storepool(self, method, mp, *args, **kwargs):
        return method(mp, store=self.store, *args, **kwargs)
