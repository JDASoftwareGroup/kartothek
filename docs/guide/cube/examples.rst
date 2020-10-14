Examples
--------
.. contents:: Table of Contents

This is a quick walk through the basic functionality of Kartothek Cubes.

First, we want to create a cube for geodata:

>>> from kartothek.core.cube.cube import Cube
>>> cube = Cube(
...     uuid_prefix="geodata",
...     dimension_columns=["city", "day"],
...     partition_columns=["country"],
... )

Apart from an abstract cube definition, we need a `simplekv`_-based storage backend:

>>> from functools import partial
>>> import tempfile
>>> import storefact
>>> store_location = tempfile.mkdtemp()
>>> store_factory = partial(
...     storefact.get_store_from_url,
...     "hfs://" + store_location,
... )
>>> store = store_factory()

Some generic setups of libraries:

>>> import pandas as pd
>>> pd.set_option("display.max_rows", 40)
>>> pd.set_option("display.width", None)
>>> pd.set_option('display.max_columns', None)
>>> pd.set_option('display.expand_frame_repr', False)

Build
`````

Kartothek cube should be initially filled with the following information:

>>> from io import StringIO
>>> import pandas as pd
>>> df_weather = pd.read_csv(
...     filepath_or_buffer=StringIO("""
... avg_temp     city country        day
...        6  Hamburg      DE 2018-01-01
...        5  Hamburg      DE 2018-01-02
...        8  Dresden      DE 2018-01-01
...        4  Dresden      DE 2018-01-02
...        6   London      UK 2018-01-01
...        8   London      UK 2018-01-02
...     """.strip()),
...     delim_whitespace=True,
...     parse_dates=["day"],
... )

We use the simple :py:mod:`kartothek.io.eager_cube` backend to store the data:

>>> from kartothek.io.eager_cube import build_cube
>>> datasets_build = build_cube(
...   data=df_weather,
...   store=store,
...   cube=cube,
... )

We just have preserved a single `Kartothek` dataset:

>>> print(", ".join(sorted(datasets_build.keys())))
seed
>>> ds_seed = datasets_build["seed"].load_all_indices(store)
>>> print(ds_seed.uuid)
geodata++seed
>>> print(", ".join(sorted(ds_seed.indices)))
city, country, day

Finally, let's have a quick look at the store content. Note that we cut out UUIDs and timestamps here for brevity.

>>> import re
>>> def print_filetree(s, prefix=""):
...     entries = []
...     for k in sorted(s.keys(prefix)):
...         k = re.sub("[a-z0-9]{32}", "<uuid>", k)
...         k = re.sub("[0-9]{4}-[0-9]{2}-[0-9]{2}((%20)|(T))[0-9]{2}%3A[0-9]{2}%3A[0-9]+.[0-9]{6}", "<ts>", k)
...         entries.append(k)
...     print("\n".join(sorted(entries)))
>>> print_filetree(store)
geodata++seed.by-dataset-metadata.json
geodata++seed/indices/city/<ts>.by-dataset-index.parquet
geodata++seed/indices/day/<ts>.by-dataset-index.parquet
geodata++seed/table/_common_metadata
geodata++seed/table/country=DE/<uuid>.parquet
geodata++seed/table/country=UK/<uuid>.parquet

Extend
``````
Now let's say we would also like to have longitude and latitude data in our cube.

>>> from kartothek.io.eager_cube import extend_cube
>>> df_location = pd.read_csv(
...     filepath_or_buffer=StringIO("""
...    city country  latitude  longitude
... Hamburg      DE 53.551086   9.993682
... Dresden      DE 51.050407  13.737262
...  London      UK 51.509865  -0.118092
...   Tokyo      JP 35.652832 139.839478
...     """.strip()),
...     delim_whitespace=True,
... )

.. hint::
    Obviously, this data does not change over time. As long as the data spans at least a single dimensions and describes
    all partition columns, you are free to use projected data for non-seed datasets.

>>> datasets_extend = extend_cube(
...   data={"latlong": df_location},
...   store=store,
...   cube=cube,
... )

This results in an extra dataset:

>>> print(", ".join(sorted(datasets_extend.keys())))
latlong
>>> ds_latlong = datasets_extend["latlong"].load_all_indices(store)
>>> print(ds_latlong.uuid)
geodata++latlong
>>> print(", ".join(sorted(ds_latlong.indices)))
country

Note that for the second dataset, no indices for ``'city'`` and ``'day'`` exists. These are only created for the seed
dataset, since that datasets forms the groundtruth about which city-day entries are part of the cube.

.. hint::
    Since the seed dataset forms the groundtruth regarding cells in the cube, additional data in other datasets are
    ignored. So in this case, ``'Tokyo'`` will be store to the cube but will cut out during queries.

If you look at the file tree, you can see that the second dataset is completely separated. This is useful to copy/backup
parts of the cube:

>>> print_filetree(store)
geodata++latlong.by-dataset-metadata.json
geodata++latlong/table/_common_metadata
geodata++latlong/table/country=DE/<uuid>.parquet
geodata++latlong/table/country=JP/<uuid>.parquet
geodata++latlong/table/country=UK/<uuid>.parquet
geodata++seed.by-dataset-metadata.json
geodata++seed/indices/city/<ts>.by-dataset-index.parquet
geodata++seed/indices/day/<ts>.by-dataset-index.parquet
geodata++seed/table/_common_metadata
geodata++seed/table/country=DE/<uuid>.parquet
geodata++seed/table/country=UK/<uuid>.parquet

Query
`````
Now the whole beauty of Kartothek Cube does not come from storing multiple datasets, but especially from retrieving the data in a
very comfortable way. It is possible to treat the entire cube as a single, large DataFrame:

>>> from kartothek.io.eager_cube import query_cube
>>> query_cube(
...     cube=cube,
...     store=store,
... )[0]
   avg_temp     city country        day   latitude  longitude
0         8  Dresden      DE 2018-01-01  51.050407  13.737262
1         4  Dresden      DE 2018-01-02  51.050407  13.737262
2         6  Hamburg      DE 2018-01-01  53.551086   9.993682
3         5  Hamburg      DE 2018-01-02  53.551086   9.993682
4         6   London      UK 2018-01-01  51.509865  -0.118092
5         8   London      UK 2018-01-02  51.509865  -0.118092

As you can see, we get a list of results back. This is because Kartothek Cube naturally supports partition-by semantic, which is
more helpful for distributed backends like `Distributed`_:

>>> dfs = query_cube(
...     cube=cube,
...     store=store,
...     partition_by="country",
... )
>>> dfs[0]
   avg_temp     city country        day   latitude  longitude
0         8  Dresden      DE 2018-01-01  51.050407  13.737262
1         4  Dresden      DE 2018-01-02  51.050407  13.737262
2         6  Hamburg      DE 2018-01-01  53.551086   9.993682
3         5  Hamburg      DE 2018-01-02  53.551086   9.993682
>>> dfs[1]
   avg_temp    city country        day   latitude  longitude
0         6  London      UK 2018-01-01  51.509865  -0.118092
1         8  London      UK 2018-01-02  51.509865  -0.118092

The query system also supports selection and projection:

>>> from kartothek.core.cube.conditions import C
>>> query_cube(
...     cube=cube,
...     store=store,
...     payload_columns=["avg_temp"],
...     conditions=(
...         (C("country") == "DE") &
...         C("latitude").in_interval(50., 52.) &
...         C("longitude").in_interval(13., 14.)
...     ),
... )[0]
   avg_temp     city country        day
0         8  Dresden      DE 2018-01-01
1         4  Dresden      DE 2018-01-02

Transform
`````````
Query and Extend can be combined to build powerful transformation pipelines. To better illustrate this we will use
`Dask.Bag`_ for that example.

.. important::
   Since `Dask`_ operations can also be executed in subprocesses, multiple threads, or even on other machines using
   `Distributed`_, Kartothek Cube requires the user to pass a :term:`Store Factory` instead of a store. This ensures that no file
   handles, TCP connections, or other non-transportable objects are shared.

>>> from kartothek.io.dask.bag_cube import (
...     extend_cube_from_bag,
...     query_cube_bag,
... )
>>> def transform(df):
...     df["avg_temp_country_min"] = df["avg_temp"].min()
...     return {
...         "transformed": df.loc[
...             :,
...             [
...                 "avg_temp_country_min",
...                 "city",
...                 "country",
...                 "day",
...             ]
...         ],
...     }
>>> transformed = query_cube_bag(
...     cube=cube,
...     store=store_factory,
...     partition_by="day",
... ).map(transform)
>>> datasets_transformed = extend_cube_from_bag(
...     data=transformed,
...     store=store_factory,
...     cube=cube,
...     ktk_cube_dataset_ids=["transformed"],
... ).compute()
>>> query_cube(
...     cube=cube,
...     store=store,
...     payload_columns=[
...         "avg_temp",
...         "avg_temp_country_min",
...     ],
... )[0]
   avg_temp  avg_temp_country_min     city country        day
0         8                     6  Dresden      DE 2018-01-01
1         4                     4  Dresden      DE 2018-01-02
2         6                     6  Hamburg      DE 2018-01-01
3         5                     4  Hamburg      DE 2018-01-02
4         6                     6   London      UK 2018-01-01
5         8                     4   London      UK 2018-01-02

Notice that the ``partition_by`` argument does not have to match the cube :term:`Partition Columns` to work. You may use
any indexed column. Keep in mind that fine-grained partitioning can have drawbacks though, namely large scheduling
overhead and many blob files which can make reading the data inefficient:

>>> print_filetree(store, "geodata++transformed")
geodata++transformed.by-dataset-metadata.json
geodata++transformed/table/_common_metadata
geodata++transformed/table/country=DE/<uuid>.parquet
geodata++transformed/table/country=DE/<uuid>.parquet
geodata++transformed/table/country=UK/<uuid>.parquet
geodata++transformed/table/country=UK/<uuid>.parquet


Append
``````
New rows can be added to the cube using an append operation:

>>> from kartothek.io.eager_cube import append_to_cube
>>> df_weather2 = pd.read_csv(
...     filepath_or_buffer=StringIO("""
... avg_temp     city country        day
...       20 Santiago      CL 2018-01-01
...       22 Santiago      CL 2018-01-02
...     """.strip()),
...     delim_whitespace=True,
...     parse_dates=["day"],
... )
>>> datasets_appended = append_to_cube(
...   data=df_weather2,
...   store=store,
...   cube=cube,
... )
>>> print_filetree(store, "geodata++seed")
geodata++seed.by-dataset-metadata.json
geodata++seed/indices/city/<ts>.by-dataset-index.parquet
geodata++seed/indices/city/<ts>.by-dataset-index.parquet
geodata++seed/indices/day/<ts>.by-dataset-index.parquet
geodata++seed/indices/day/<ts>.by-dataset-index.parquet
geodata++seed/table/_common_metadata
geodata++seed/table/country=CL/<uuid>.parquet
geodata++seed/table/country=DE/<uuid>.parquet
geodata++seed/table/country=UK/<uuid>.parquet

Notice that the indices where updated automatically.

>>> query_cube(
...     cube=cube,
...     store=store,
... )[0]
   avg_temp  avg_temp_country_min      city country        day   latitude  longitude
0         8                   6.0   Dresden      DE 2018-01-01  51.050407  13.737262
1         4                   4.0   Dresden      DE 2018-01-02  51.050407  13.737262
2         6                   6.0   Hamburg      DE 2018-01-01  53.551086   9.993682
3         5                   4.0   Hamburg      DE 2018-01-02  53.551086   9.993682
4         6                   6.0    London      UK 2018-01-01  51.509865  -0.118092
5         8                   4.0    London      UK 2018-01-02  51.509865  -0.118092
6        20                   NaN  Santiago      CL 2018-01-01        NaN        NaN
7        22                   NaN  Santiago      CL 2018-01-02        NaN        NaN

Remove
``````
You can remove entire partitions from the cube using the remove operation:

>>> from kartothek.io.eager_cube import remove_partitions
>>> datasets_after_removal = remove_partitions(
...     cube=cube,
...     store=store,
...     ktk_cube_dataset_ids=["latlong"],
...     conditions=(C("country") == "UK"),
... )
>>> query_cube(
...     cube=cube,
...     store=store,
... )[0]
   avg_temp  avg_temp_country_min      city country        day   latitude  longitude
0         8                   6.0   Dresden      DE 2018-01-01  51.050407  13.737262
1         4                   4.0   Dresden      DE 2018-01-02  51.050407  13.737262
2         6                   6.0   Hamburg      DE 2018-01-01  53.551086   9.993682
3         5                   4.0   Hamburg      DE 2018-01-02  53.551086   9.993682
4         6                   6.0    London      UK 2018-01-01        NaN        NaN
5         8                   4.0    London      UK 2018-01-02        NaN        NaN
6        20                   NaN  Santiago      CL 2018-01-01        NaN        NaN
7        22                   NaN  Santiago      CL 2018-01-02        NaN        NaN

Delete
``````
You can also delete entire datasets (or the entire cube):

>>> from kartothek.io.eager_cube import delete_cube
>>> datasets_still_in_cube = delete_cube(
...     cube=cube,
...     store=store,
...     datasets=["transformed"],
... )
>>> query_cube(
...     cube=cube,
...     store=store,
... )[0]
   avg_temp      city country        day   latitude  longitude
0         8   Dresden      DE 2018-01-01  51.050407  13.737262
1         4   Dresden      DE 2018-01-02  51.050407  13.737262
2         6   Hamburg      DE 2018-01-01  53.551086   9.993682
3         5   Hamburg      DE 2018-01-02  53.551086   9.993682
4         6    London      UK 2018-01-01        NaN        NaN
5         8    London      UK 2018-01-02        NaN        NaN
6        20  Santiago      CL 2018-01-01        NaN        NaN
7        22  Santiago      CL 2018-01-02        NaN        NaN

Dimensionality and Partitioning
```````````````````````````````
Sometimes, you have data that only exists in a projection of the cube, like the ``latlong`` data from the `Extend`_
section. For non-seed datasets, you can just leave out :term:`Dimension Columns`, as long as at least a single
:term:`Dimension Column` remains.

Sometimes, you may find that the standard partitioning does not match the data really well, so for non-seed datasets, you can change the partitioning:

- **leave out partition columns:** especially helpful when the dataset is really small or data only exists on a specific
  projection that does lead to partitioning (e.g. the ``day`` dimension from the example cube)
- **additional partition columns:** when the dataset has many and/or very memory-intense columns

.. important::

    Although other partitionings than the cube :term:`Partition Columns` can be specified, it is strongly adviced to not
    diverge too much from these for performance reasons.

>>> df_time = pd.DataFrame({
...     "day": pd.date_range(
...         start="2018-01-01",
...         end="2019-01-01",
...         freq="D",
...     ),
... })
>>> df_time["weekday"] = df_time.day.dt.weekday
>>> df_time["month"] = df_time.day.dt.month
>>> df_time["year"] = df_time.day.dt.year
>>> datasets_time = extend_cube(
...   data={"time": df_time},
...   store=store,
...   cube=cube,
...   partition_on={"time": []},
... )
>>> print_filetree(store, "geodata++time")
geodata++time.by-dataset-metadata.json
geodata++time/table/<uuid>.parquet
geodata++time/table/_common_metadata
>>> query_cube(
...     cube=cube,
...     store=store,
... )[0]
   avg_temp      city country        day   latitude  longitude  month  weekday  year
0         8   Dresden      DE 2018-01-01  51.050407  13.737262      1        0  2018
1         4   Dresden      DE 2018-01-02  51.050407  13.737262      1        1  2018
2         6   Hamburg      DE 2018-01-01  53.551086   9.993682      1        0  2018
3         5   Hamburg      DE 2018-01-02  53.551086   9.993682      1        1  2018
4         6    London      UK 2018-01-01        NaN        NaN      1        0  2018
5         8    London      UK 2018-01-02        NaN        NaN      1        1  2018
6        20  Santiago      CL 2018-01-01        NaN        NaN      1        0  2018
7        22  Santiago      CL 2018-01-02        NaN        NaN      1        1  2018


.. _Distributed: https://distributed.readthedocs.io/
.. _DataFrame.merge: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.merge.html?highlight=merge#pandas.DataFrame.merge
.. _DataFrame.reset_index: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.reset_index.html?highlight=reset_index#pandas.DataFrame.reset_index
.. _Dask: https://docs.dask.org/
.. _Dask.Bag: https://docs.dask.org/en/latest/bag.html
.. _Dask.DataFrame: https://docs.dask.org/en/latest/dataframe.html
.. _simplekv: https://simplekv.readthedocs.io/
