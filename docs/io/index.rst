.. _input_output:

==============
In- and Output
==============

The :mod:`kartothek.io_components` module offers a collection of building
blocks to assemble data pipelines to read, write or merge Kartothek datasets.
All pipeline blocks ensure compliance with the dataset specification in
:ref:`dataset_spec`.

Since a central focus of the dataset specification includes portability, this
module offers multiple backend implementations suited for different use cases,
e.g. streaming, bulk processing or fast small data batch retrieval.

All pipelines are designed in such a way as to avoid network transmission
wherever possible (unless explicitly requested) such that the pipelines may run
on remote schedulers which do not offer managed data locality or
worker-to-worker communication. This allows the most efficient, parallelized
pipeline execution without the need for advanced query optimization.

The module is built in such a way that other implementations may re-use the
core pipeline building blocks and use them to implement the same pipelines in
a different framework.

Currently included backend implementations can be found in their respective
submodules:

:mod:`~kartothek.io.eager`
--------------------------

    Immediate pipeline execution on a single worker without the need for any
    external scheduling engine. Well suited for small data, low-overhead
    pipeline execution.

:mod:`~kartothek.io.iter`
-------------------------

    An iteration interface implementation as python generators to allow for
    (partition based) stream / micro-batch processing of data.

:mod:`~kartothek.io.dask`
-------------------------

    The `dask` module offers a seamless integration to `dask <https://docs.dask.org>`_
    and offers implementations for dask data collections like `dask.Bag`,
    `dask.DataFrame` or as `dask.Delayed`.
    This implemenation is best suited to handle big data and scale the
    pipelines across many workers using `dask.distributed`.

.. include:: examples.rst
