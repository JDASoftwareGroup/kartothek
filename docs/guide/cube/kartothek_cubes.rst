===============
Kartothek Cubes
===============

Imagine a typical machine learning workflow, which might look like this:

    - First, we get some input data, or source data. In the context of Kartothek cubes, we will refer to the source data as seed data or seed dataset.
    - On this seed dataset, we might want to train a model that generates predictions.
    - Based on these predicitons, we might want to generate reports and calculate KPIs.
    - Last, but not least, we might want to create some dashboards showing plots of the aggregated KPIs as well as the underlying input data.

What we need for this workflow is not a table-like view on our data, but a single (virtual) view on everything that we generated in these different steps.

Kartothek Cubes deal with multiple `Kartothek` datasets loosely modeled after `Data Cubes`_.

One cube is made by multiple `Kartothek` datasets. User-facing APIs are mostly consume and provide `Pandas`_ DataFrames.

Cubes offer an interface to query all of the data without performing complex join operations manually each time.
Because kartothek offers a view on our cube similar to large virtual pandas DataFrame, querying the whole dataset is very comfortable.


.. _Data Cubes: https://en.wikipedia.org/wiki/Data_cube
.. _Pandas: https://pandas.pydata.org/