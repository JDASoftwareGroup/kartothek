.. _storage_layout:

==============
Storage Layout
==============

Kartothek structures your data using these concepts:

- One whole unit of data that Kartothek manages is called a *dataset*.
- A dataset consists of one or more *tables* that each have a *schema*. 
- Table rows are partitioned by any number of columns: Rows having the same combination
  of values in these columns are grouped together.
- A partition consists of one or more Parquet files, which contain a chunk of rows that
  were written at a time.
- Kartothek can also generate an index for any number of columns, which speeds up finding
  the relevant Parquet files for specific values for the indexed column.

A general Kartothek storage layout thus looks as follows::

  ─ <dataset_uuid>.by-dataset-metadata.json
  ─ <dataset_uuid>/
    ├── <table1>/
    │   ├── _common_metadata
    │   ├── <partition1>=value/
    │   │   ├── <partition2>=value/
    │   │   │   ├ ...
    │   │   │       ├── <partitionN>=value/
    │   │   │       │   ├── df1.parquet
    │   │   │       │   ├── df2.parquet
    │   │   │       │   └── ...
    │   │   │       ├── <partitionN>=value/
    │   │   │       │   ├── df1.parquet
    │   │   │       │   ├── df2.parquet
    │   │   │       │   └── ...
    │   │   │       └── ...
    │   │   ├── <partition2>=value/
    │   │   │   ├ ...
    │   │   │       ├── <partitionN>=value/
    │   │   │       │   ├── df1.parquet
    │   │   │       │   ├── df2.parquet
    │   │   │       │   └── ...
    │   │   │       ├── <partitionN>=value/
    │   │   │       │   ├── df1.parquet
    │   │   │       │   ├── df2.parquet
    │   │   │       │   └── ...
    │   │   │       └── ...
    │   │   └── <partition2>=value/ ...
    │   ├── <partition1>=value/ ...
    │   └── <partition1>=value/ ...
    ├── <table2>/ ...
    ├── <table3>/ ...
    └── indices/
        ├── <index_column1>/
        │   └── <timestamp>.by-dataset-index.parquet
        ├── <index_column2>/ ...
        └── <index_column3>/ ...

Where:

- ``<dataset_uuid>.by-dataset-metadata.json`` contains the ``DatasetMetadata`` you have seen above.
- ``<tableN>`` contains the data for any tables in the dataset, partitioned by N >= 0 columns. The directory structure will be N folders deep.
- ``_common_metadata`` contains the table schema of ``dfN.parquet``. It is always identical for all Parquet files of a table.
- ``indices`` contains a database index for each index column, for quick lookup of rows where the column value matches a given value.