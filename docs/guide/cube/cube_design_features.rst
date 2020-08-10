

Design
******

This section discusses different design aspects of kartothek Cubes Functionality. Not that many decisions here are heavily bound to the
`Kartothek` format and the features it provides. Also, Kartothek Cube must support the usage patterns we have seen and
provide a clear API (in technical and semantical terms). It especially is no allrounder solution and we do NOT want to
support all possible use cases.

.. important::
    Kartothek Cube is an opinionated framework!

Multi-Dataset VS Multi-Table
============================
When mapping multiple parts (tables or datasets) to `Kartothek`, we have the following options:

a) *Multi-Table:* As used by Klee1, use a single dataset w/ multiple tables.
b) **Multi-Dataset:** Use multiple datasets each w/ a single table.

We use multiple `Kartothek` datasets instead of multiple tables (like Klee1 did) for the following reasons:

- `Kartothek` discourages unbalanced tables, e.g. when Kartothek Cube writes a seed table and an enrich step is only done for a
  part of the data (e.g. the present and future). This worked for Klee1 because we have used a way simpler format
  (based on Kartothek V2) and ignored the recommendation of the `Kartothek` main author.
- OSS tools like `Dask.DataFrame`_ do not support any form of multi-table datasets and the current `Kartothek` format
  is just a workaround to make multi-table datasets look like multiple regular datasets.
- The `Kartothek`developers want to deprecate multi-table `Kartothek` datasets in the future.
- `Kartothek` needs a central metadata file for performance reason (listing blob containers w/ massive amounts of blobs
  is very slow) which is shared between tables. So running 2 tasks that create 2 completely unrelated tables would still
  result in write conflicts. Using separate datasets solves this issue.
- Using multiple datasets allows users to copy, backup and delete them separately.
- Index structures are bound to datasets which feels more consistent than the former solution where you did not know
  which table was meant by an index entry.

Implicit Existence VS Explicit Cube Metadata
============================================
There are two ways how a cube could be created out of `Kartothek` datasets:

a) **UUID Namespace:** specify a UUID prefix and the base cube on existing of datasets (if a dataset w/ a given prefix
   exists, it is part of the cube)
b) *Cube Metadata File:* write another metadata file, similar to what `Kartothek` is doing but like a "dataset of
   datasets"

By using another metadata file, we would introduce a potential write conflict, one that was solved by using multiple
datasets instead of multiple tables. Also, it would make copying, deleting and backup up datasets in an independent
fashion more difficult.

Using an UUID prefix an implicit existing does not has these issues, but requires us to list the store content for
dataset discovery. While listing all store keys can be really slow, listing w/ a path separator is really quick,
especially if the number of "folders" and "top-level files" is small, which is given using `Kartothek` V4 datasets.
Also, it allows us to load external datasets (like ones produces by other systems) as cubes and therefore simplifies
interoperability.

.. important::
    Since all datasets that match a given prefix are part of the cube, they all must be valid in terms of the `Format`_.

Explicit Physical Partitions
============================
There are multiple ways to store data to the blob store:

a) *Single, Large Parquet Files:* Just one giant `Parquet`_ file for everything.
b) **Explicit Physical Partitions:** Split the data into partitions described by dedicated columns.
c) *Automatic Partitioning:* Leave the load-balancing entirely to kartothek_cube.

We have decided for explicit physical partitions since we have seen that this data model works well for our current data
flow. It allows quick and efficient re-partitioning to allow row-based, group-based, and timeseries-based data
processing steps, while keeping the technical complexity rather low (compared to an automatic + dynamic partitioning).
It also maps well to multiple backends we plan to use.

Using a single, large `Parquet`_ file would scale during the read path due to predicate pushdown and using well-thought
chunk sizes, but either require a write path that supports writing individual chunks (currently not available and
unlikely to be supported anytime soon) or one fat-node at the end of every write pipeline which is inefficient and even
impossible w/ our current architecture.

In a future version of Klee, there may be a way to get the data automatically partitioned, maybe even w/ some
feedback-loop based approach (submit tasks, watch memory/CPU consumption, adjust, re-submit).

.. important::
    Note that while physical partitions are explicit, their semantic impact should be small. They are an optimization
    and may speed up load and store operations, but cubes w/ different partition columns but built out of the same data
    should behave the same (except some query features like re-partioning may be differ due to missing indices).

Timestamp Column and Update/Overwrite Semantic
==============================================
There are cases where instead of building a new cube or extending it with new datasets, users want to change existing
datasets in one of the following ways:

- appending new data (i.e., new rows)
- overwriting existing data (e.g. due to retro-active changes or code updates)
- removing data (e.g. because data should be masked out)
- a mixture of these

`Kartothek` discourages overwriting partitions on the fly because this would disallow the implementation of a
commit-semantic (see all or no changes). Therefore, we have two options:

a) **Deletion / Overwrite Masks:** Specifying some kind of "deletion/overwrite mask" during update operations that tell
   Kartothek_Cube + `Kartothek` which parts of the cube will be changed.
b) *Timestamp-based:* Use update timestamps and an append-only approach, similar to `Event Sourcing`_.

Kartothek Cube only supports Deletion / Overwrite Masks.

Deletion / overwrite masks are certainly easier to implement and are out-of-the-box-compatible with `Hive`_, but we
often encountered the situation where it would have been nice to recover an old cube state. Update timestamps allow
this, but require special handling by the query system.

Update Granularity
==================
We are aware of two ways to design the update granularity:

a) **Partition-wise:** Entire partitions can overwrite old physical partitions. Deletion operations are partition-wise.
b) *Row-wise:* The entire cube behaves like one large, virtual DataFrame and the user handles rows. Physical partitions
   are just an optimization.

While the row-wise approach has the nice property that the physical partitioning is just an optimization, it is complex
to implement and a major performance problem, especially when many transaction were written to the cube. This is due to
the fact that the query planner cannot inspect the cube cells from each parquet file without reading it and therefore
either needs a very expensive planning phase with loads of IO operations or it cannot prune data early, leading to
massive IO in the execution phase. So we decided for partition-wise IO, which turned out to be quite simple to
implement.

Causality Tracking
==================
Imagine the following scenario:

1. Cube is built with a single seed dataset from a Database.
2. Extra dataset is added bases in "seed".
3. Seed dataset is updated from the Database (columns removed/appended/overwritten)

Since the extra dataset was calculated with data from "seed", causality flows from "seed" to "extra". If the cube is now
queried, it is questionable if new "seed" data should or should not be joined to old "extra" data. Kartothek Cube currently does
NOT track causality and leaves it to the user to solve this issue, since there might be good reasons for both solutions
and we (the Kartothek Cube authors) currently do not have enough experience to judge which way should be strictly preferred.

A possible solution could use the extra metadata of every dataset and a `Vector Clock`_ to track causality.

Column Namespacing
==================
There are multiple options regarding the mapping of dataset columns to DataFrame columns:

a) **No Mapping:** do not change column names but prohibit collisions of payload columns (i.e. columns that are neither
   dimension nor partition columns)
b) *Namespace all Columns:* e.g. dimension and partition columns have now prefix, but payload columns have the form
   ``<dataset ID>__<column name>``
c) *Namespace on Demand:* only prefix in case of a name collision, similar to what `DataFrame.merge`_ is doing.

Since Kartothek Cube is intended to use in production, "Namespace on Demand" is not an option since it may result in hard to
debug runtime errors. "Namespace all Columns" is a very stable option, but would require every part of our data
processing pipelines to know which dataset produces which column. Currently, this is transparent and columns can be
moved from one stage to another w/o resulting to larger code changes. We would like to keep this nice behavior, so we
went for "No Mapping".

Hardcoded Format Parameters
===========================
When it comes to format parameters like "Storage Format == Parquet", there are basically three options:

a) *Fully Configurable, No Defaults:* Just leave the decision to the user.
b) *Configurable w/ Defaults:* Pre-configure all options but allow the user to overwrite things (similar to `Kartothek`
   and Klee1).
c) **Hardcoded Parameters:** Keep everything hardcoded, no choice for the user.

Fully configurable parameters w/o any defaults would make the API complex, the test suite large, would require us to
think about many partly absurd edge cases, while most of our users actually do not care at all. They just want something
that works.

Defaults w/ configurable options still makes the API complex (even though less by default, because users only need to
describe divergence from the standard), but we have seen no good applications of config options in Klee1. Instead, it
has led to arbitrary choices and suboptimal configuration. Also, it resulted in a test suite that covered the defaults
well but basically left out most config options, ultimately exposing a dangerous API to the user.

So we have decided for hardcoded parameters because we do not want to deal with the complexity and think we know better
than most of our users.

Seed-Based Join System
======================
When data is stored in multiple parts (tables or datasets), the question is how to expose it to the user during read
operations:

a) *Seperate DataFrames:* Conditions are group-by operations are applied to the individual parts and no join operations
   are performed by kartothek_cube.
b) **Seed-Based Join:** Mark a single part as seed which provides the groundtruth regarding cells (i.e. unique dimension
   entries) in the cube, all other parts are just additional columns.
c) *Fully Configurable Join Order:* Leave it to the user to configure the join order (this was done in an early version
   of Klee1).

Separate DataFrames would give the user full control, but would also force them to create load of boilerplate code,
likely resulting in another framework on top of kartothek_cube. This would contradict any `KISS`_ approach we try to take here.
Also it makes reasoning about conditions and partition-by parameters more difficult since it is not always clear how
these effects cross-influence different parts of the cube.

Using a fully configurable was tried in Klee1, but it turned out that many users do not want to care about the
consequences of this complex beast. Also, it makes predicate pushdown and efficient index operations more difficult to
implement, especially since the core of Kartothek Cube is based on `Pandas`_ which lacks proper NULL-handling.

Finally, we have decided for a seed-based system some time ago in Klee1 and our users are happy and know what to expect.
It is also easy to teach, good to implement and test, and it matches the semantic of our data processing pipelines
(groundtruth from an initial external source, subsequent enrichments w/ additional columns on top of it.)

.. important::
    There are two variants of the seed-based system:

    a) *Enforced:* Cells in non-seed datasets must be present in the seed dataset. This is enforced during write
       operations.
    b) **Lazy:** The seed semantic is only enforced during queries.

    We have decided for the lazy approach, since it better supports independent copies and backups of datasets and also
    simplifies some of our processing pipelines (e.g. geolocation data can blindly be fetched for too many locaations and dates.)


Format
******

This section describes how `Kartothek`  must be structured to be consumed by kartothek_cube.

Cube
====
An abstract cube is described by the following attributes:

- **UUID Prefix:** A common prefix for UUIDs of all datasets that are part of the cube.
- **Dimension Columns:** Which orthogonal dimensions form the cube. Hence, every cell described by these columns is
  unique.
- **Partition Columns:** Columns that describe how data is partitioned when written to the blob store. These columns
  will form the `Kartothek`  ``partition_on`` attribute.
- **Seed Dataset:** Which dataset forms the ground truth regarding the set of cells in the cube.
- **Index Columns:** Which non-dimension and non-partition columns should also be indexed.

Datasets
========
All `Kartothek`  datasets that are part of a cube must fulfill the following criteria:

- **Kartothek Dataset UUID:** ``'<UUID Prefix>++<Dataset UUID>'``. E.g. for a cube called ``'my_cube'`` and a dataset
  called ``'weather'``, the UUID that is used in `Kartothek`  is ``'my_cube++weather'``.
- **Metadata Version:** 4
- **Metadata Storage Format:** `JSON`_ (`MessagePack`_ can be read as well)
- **DataFrame Serialization Format:** `Parquet`_ with `ZSTD`_ compression (other compressions can be read as well)
- **Kartothek Tables:** Only a single one called ``'table'`` (same as ``SINGLE_TABLE`` in `Kartothek`)
- **Partition Keys:**

  - **Seed Dataset:** ``<Partition Columns>``.
  - **Other Datasets:** Can be anything.

- **Partition Labels:** The user has no ability set the partition labels, instead the default `Kartothek` `UUID4`_
  generation mechanism is used.

Indices
=======
The following index structures must be present (additional indices will be ignored):

- **Partition Indices:** According to the partition keys described above.
- **Explicit Secondary Indices:** For all index columns. For the seed dataset also for all dimension columns. Additional
  indices may exist and can be used by the query engine.

Metadata
========
Kartothek Cube allows the user to specify per-dataset metadata. Furthermore, the following entries are added by default to every
dataset:

- ``'ktk_cube_is_seed'``: boolean entry to mark the seed dataset
- ``'ktk_cube_dimension_columns'``: list of :term:`Dimension Columns`
- ``'ktk_cube_partition_columns'``: list of :term:`Partition Columns`

DataFrame Normalization
=======================
On top of what `Kartothek`  is doing, the following properties of preserved DataFrames will be ensured:

- all column names are unicode strings (``str``); that especially excludes integer-based column names
- DataFrame indices are range indices starting at 0 with a step size of 1; this is equivalent to
  `DataFrame.reset_index`_
- values are sorted by dimension columns (if present) in the order given by cube specification



Dask DataFrame / Hive Interoperability
**************************************

.. warning::

   Due to bugs in ``pyarrow`` and ``fsspec``, this is currently broken!

The single datasets produced by Kartothek Cube should be compatible with `Hive`_ and `Dask.DataFrame`_:

.. >>> import dask.dataframe as ddf
.. >>> hive_location = '{}/{}/table'.format(
.. ...     store_location,
.. ...     cube.ktk_dataset_uuid(cube.seed_dataset),
.. ... )
.. >>> df_dask = ddf.read_parquet(hive_location)
.. >>> df_dask.npartitions
.. 2
.. >>> round(df_dask['avg_temp'].mean().compute(), 2)
.. 6.17

Keep in mind that only the single datasets, not the entire cube can be read directly with `Dask`_. To read the whole cube, you must implement either the query code yourself (follow the :ref:`query_system`  as a guide) or use the functionality in :py:mod:`kartothek.io.dask.dataframe_cube` that provides a `Dask.DataFrame`_ interface but does not use `Dask`_ for the direct IO.

.. important::
   Even though we try to ensure compatbility with other implementations, we cannot guarantee that. We have seen many
   cases in the past where parquet files are processed in incompatible ways or where certain data types are not
   supported. In case you find such a problem, you may report this to the Kartothek developers so we either can fix this on
   our side, their side, or record this as a known issue.

Documentation
=============
Examples in docstrings, README and specs should use real-world column names (like ``'COUNTRY_CODE'``).

Examples in pytest should use simplified column names:

- dimension columns: ``'x'``, ``'y'``,...
- partition columns: ``'p'``, ``'q'``,...
- index columns: ``'i1'``, ``'i2'``,...
- payload columns: ``'v1'``, ``'v2'``,...

CLI examples are produced using ``kartothek_cube --color=always ... | terminal-to-html > out.html`` with `terminal-to-html`_ and
are wrapped into the following code snipped:

.. code-block:: rst

   .. raw:: html

      <pre>
      ...
      </pre>

Glossary
********

.. glossary::

    Build
        Process of creating a new cube.

    Cell
        A unique combination of :term:`Dimension` values. Will result in a single row in input and output DataFrames.

    Cube
        A combination of multiple datasets that model an `Data Cubes`_-like construct. The core data structure of kartothek cube.

    Dataset ID
        The ID of a dataset that belongs to the cube w/o any :term:`UUID prefix`.

    Dimension
        Part of the address for a certain cube :term:`Cell`. Usually refered as :term:`Dimension Column`. Different
        dimension should describe orthogonal attributes.

    Dimension Column
        DataFrame column that contains values for a certain :term:`Dimension`.

    Dimension Columns
        Ordered list of all :term:`Dimension Column` for a :term:`Cube`.

    Extend
        Process of adding new datasets to an existing cube.

    Index Column
        Column for which additional index structures are build.

    Kartothek Dataset UUID
        Name that makes a dataset unique in a store, includes :term:`UUID prefix` and :term:`Dataset ID` as
        ``<UUID Prefix>++<Dataset ID>``.

    Logical Partition
        Partition that was created by ``partition_by`` arguments to the :term:`Query`.

    Physical Partition
        A single chunk of data that is stored to the blob store. May contain multiple `Parquet`_ files.

    Partition Column
        DataFrame column that contains one part that makes a :term:`Physical Partition`.

    Partition Columns
        Ordered list of all :term:`Partition Column` for a :term:`Cube`.

    Projection
        Process of dimension reduction of a cube (like a 3D object projects a shadow on the wall). Only works if the
        involved payload only exists in the subdimensional space since no automatic aggregation is supported.

    Seed
        Dataset that provides the groundtruth about which :term:`Cell` are in a :term:`Cube`.

    Store Factory
        A callable that does not take any arguments and creates a new `simplekv`_ store when being called. Its type is
        ``Callable[[], simplekv.KeyValueStore]``.

    Query
        A request for data from the cube, including things like "payload columns", "conditions", and more.

    Query Execution
        Process of reading out data from a :term:`Cube`, aka the execution of a :term:`Query`.

    Query Intention
        The actual intention of a :term:`Query`, e.g.:

        - if the user queries "all columns", the intention includes the concrete set of columns
        - if the user does not specify the dimension columns, it should use the cube dimension column (aka "no
          :term:`Projection`")

    UUID Prefix
        Common prefix for all datasets that belong to a :term:`Cube`.


.. _Blue Yonder: https://www.blueyonder.com/
.. _CSV: https://en.wikipedia.org/wiki/Comma-separated_values
.. _Data Cubes: https://en.wikipedia.org/wiki/Data_cube
.. _DataFrame.merge: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.merge.html?highlight=merge#pandas.DataFrame.merge
.. _DataFrame.reset_index: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.reset_index.html?highlight=reset_index#pandas.DataFrame.reset_index
.. _Dask: https://docs.dask.org/
.. _Dask.Bag: https://docs.dask.org/en/latest/bag-overview.html
.. _Dask.DataFrame: https://docs.dask.org/en/latest/dataframe.html
.. _Event Sourcing: https://martinfowler.com/eaaDev/EventSourcing.html
.. _Hive: https://hive.apache.org/
.. _JSON: https://json.org/
.. _KISS: https://en.wikipedia.org/wiki/KISS_principle
.. _MessagePack: https://msgpack.org/
.. _Pandas: https://pandas.pydata.org/
.. _Parquet: https://parquet.apache.org/
.. _PostgreSQL: https://www.postgresql.org/
.. _Presto: https://prestodb.io/
.. _simplekv: https://simplekv.readthedocs.io/
.. _terminal-to-html: https://github.com/buildkite/terminal-to-html
.. _UUID4: https://en.wikipedia.org/wiki/Universally_unique_identifier#Version_4_(random)
.. _Vector Clock: https://en.wikipedia.org/wiki/Vector_clock
.. _ZSTD: https://github.com/facebook/zstd
