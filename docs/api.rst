API
===

This is a non-exhaustive list of the most useful kartothek functions.

Please see :ref:`versioning` for guarantees we currently provide for the stability of the interface.

Dataset state and metadata
--------------------------

.. currentmodule:: kartothek.core


Core functions and classes to investigate the dataset state.

.. autosummary::

    dataset.DatasetMetadata
    factory.DatasetFactory
    common_metadata.SchemaWrapper


Data retrieval and storage
--------------------------

Eager
*****

.. currentmodule:: kartothek.io.eager

Immediate pipeline execution on a single worker without the need for any
external scheduling engine. Well suited for small data, low-overhead
pipeline execution.

High level user interface
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::

    read_table
    read_dataset_as_dataframes
    store_dataframes_as_dataset
    update_dataset_from_dataframes
    build_dataset_indices
    garbage_collect_dataset
    delete_dataset

Expert low level interface
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::

    read_dataset_as_metapartitions
    create_empty_dataset_header
    write_single_partition
    commit_dataset

Iter
****

.. currentmodule:: kartothek.io.iter

An iteration interface implementation as python generators to allow for
(partition based) stream / micro-batch processing of data.

High level user interface
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::

    read_dataset_as_dataframes__iterator
    update_dataset_from_dataframes__iter
    store_dataframes_as_dataset__iter

Expert low level interface
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::

    read_dataset_as_metapartitions__iterator


Dask
****

The `dask` module offers a seamless integration to `dask <https://docs.dask.org>`_
and offers implementations for dask data collections like `dask.Bag`,
`dask.DataFrame` or as `dask.Delayed`.
This implementation is best suited to handle big data and scale the
pipelines across many workers using `dask.distributed`.

DataFrame
^^^^^^^^^

This is the most user friendly interface of the dask containers and offers direct access to the dask DataFrame.

.. currentmodule:: kartothek.io.dask.dataframe


.. autosummary::

    read_dataset_as_ddf
    store_dataset_from_ddf
    update_dataset_from_ddf
    collect_dataset_metadata
    hash_dataset

.. currentmodule:: kartothek.io.dask.compression

.. autosummary::

    pack_payload_pandas
    pack_payload
    unpack_payload_pandas
    unpack_payload

Bag
^^^

This offers the dataset as a dask Bag. Very well suited for (almost) embarassingly parallel batch processing workloads.

.. currentmodule:: kartothek.io.dask.bag


.. autosummary::

    read_dataset_as_dataframe_bag
    store_bag_as_dataset
    build_dataset_indices__bag


Delayed
^^^^^^^

This offers a low level interface exposing the delayed interface directly.

.. currentmodule:: kartothek.io.dask.delayed


.. autosummary::

    read_table_as_delayed
    read_dataset_as_delayed
    store_delayed_as_dataset
    update_dataset_from_delayed
    merge_datasets_as_delayed
    delete_dataset__delayed
    garbage_collect_dataset__delayed


DataFrame Serialization
-----------------------

.. currentmodule:: kartothek.serialization

DataFrame serializers
*********************

.. autosummary::

    DataFrameSerializer
    CsvSerializer
    ParquetSerializer


Utility to handle predicates
****************************

.. autosummary::

    filter_predicates_by_column
    columns_in_predicates
    filter_df_from_predicates
    filter_array_like
