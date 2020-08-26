
Query System
------------
.. contents:: Table of Contents

Kartothek views the whole cube as a large, virtual DataFrame. The seed dataset presents the groundtruth regarding rows, all
other datasets are joined via a left join. The user should not see that data is partitioned via
:term:`Partition Columns` or split along datasets.

.. important::
    It is a common misconception that Kartothek is able to join arbitrary datasets or implements a complete join system like
    SQL. This is NOT the case!

This section explain some technical details around this mechanism.

Per-Dataset Partitions
``````````````````````
First of all, all partition files for all datasets are gathered. Every partition file is represented by a unique label.
For every dataset, index data for the Primary Indices (aka partition columns) will be loaded and joined w/ the labels::

     P | Q | __ktk_cube_labels_seed
    ===+===+=========================
     0 | 0 | P=0/Q=0/<uuid1>.parquet
     0 | 1 | P=0/Q=1/<uuid2>.parquet
     0 | 1 | P=0/Q=1/<uuid3>.parquet
     1 | 0 | P=1/Q=0/<uuid4>.parquet
     1 | 0 | P=1/Q=1/<uuid5>.parquet


Also, pre-conditions are applied during that step. These are conditions that can be evaluated based on index data
(Partition Indices, Explicit Secondary Indices for dimension columns as well as index columns)::

    condition = (I1 > 1)

    join index information:

     P | Q | I1 | __ktk_cube_labels_seed
    ===+===+====+=========================
     0 | 0 |  1 | P=0/Q=0/<uuid1>.parquet
     0 | 1 |  2 | P=0/Q=1/<uuid2>.parquet
     0 | 1 |  3 | P=0/Q=1/<uuid3>.parquet
     1 | 0 |  4 | P=1/Q=0/<uuid4>.parquet
     1 | 0 |  5 | P=1/Q=1/<uuid5>.parquet

     filter:

     P | Q | I1 | __ktk_cube_labels_seed
    ===+===+====+=========================
     0 | 1 |  2 | P=0/Q=1/<uuid2>.parquet
     0 | 1 |  3 | P=0/Q=1/<uuid3>.parquet
     1 | 0 |  4 | P=1/Q=0/<uuid4>.parquet
     1 | 0 |  5 | P=1/Q=0/<uuid5>.parquet

     remove index information:

     P | Q | __ktk_cube_labels_seed
    ===+===+=========================
     0 | 1 | P=0/Q=1/<uuid2>.parquet
     0 | 1 | P=0/Q=1/<uuid3>.parquet
     1 | 0 | P=1/Q=0/<uuid4>.parquet
     1 | 0 | P=1/Q=1/<uuid5>.parquet


Now, partition-by data is added (if not already present)::

    partition-by = I2

     P | Q | I2 | __ktk_cube_labels_seed
    ===+===+====+=========================
     0 | 1 |  1 | P=0/Q=1/<uuid2>.parquet
     0 | 1 |  1 | P=0/Q=1/<uuid3>.parquet
     1 | 0 |  1 | P=1/Q=0/<uuid4>.parquet
     1 | 0 |  2 | P=1/Q=0/<uuid5>.parquet

Finally, rows w/ identical partition information (physical and partition-by) are compactified::

     P | Q | I2 | __ktk_cube_labels_seed
    ===+===+====+==================================================
     0 | 1 |  1 | P=0/Q=1/<uuid2>.parquet, P=0/Q=1/<uuid3>.parquet
     1 | 0 |  1 | P=1/Q=0/<uuid4>.parquet
     1 | 0 |  2 | P=1/Q=0/<uuid5>.parquet


Alignment
`````````
After data is prepared for every dataset, they are aligned using their physical partitions. Partitions that are present
in non-seed datasets but are missing from the seed dataset are dropped::

    inputs:

     P | Q | I2 | __ktk_cube_labels_seed
    ===+===+====+==================================================
     0 | 1 |  1 | P=0/Q=1/<uuid2>.parquet, P=0/Q=1/<uuid3>.parquet
     1 | 0 |  1 | P=1/Q=0/<uuid4>.parquet
     1 | 0 |  2 | P=1/Q=0/<uuid5>.parquet

     P | Q | __ktk_cube_labels_enrich
    ===+===+==================================================
     0 | 0 | P=0/Q=1/<uuid6>.parquet
     0 | 1 | P=0/Q=1/<uuid7>.parquet
     1 | 0 | P=1/Q=0/<uuid8>.parquet, P=0/Q=1/<uuid9>.parquet
     9 | 0 | P=9/Q=0/<uuid0>.parquet


     output:

     P | Q | I2 | __ktk_cube_labels_seed                               | __ktk_cube_labels_enrich
    ===+===+====+==================================================+==================================================
     0 | 1 |  1 | P=0/Q=1/<uuid2>.parquet, P=0/Q=1/<uuid3>.parquet | P=0/Q=1/<uuid7>.parquet
     1 | 0 |  1 | P=1/Q=0/<uuid4>.parquet                          | P=1/Q=0/<uuid8>.parquet, P=0/Q=1/<uuid9>.parquet
     1 | 0 |  2 | P=1/Q=0/<uuid5>.parquet                          | P=1/Q=0/<uuid8>.parquet, P=0/Q=1/<uuid9>.parquet


In case pre-conditions got applied to any non-seed dataset or partition-by columns that are neither a
:term:`Partition Column` nor :term:`Dimension Column`, the resulting join will be an inner join. This may result in
removing potential partitions early.

Re-grouping
```````````
Now, the DataFrame is grouped by partition-by::

    partition-by: I2

    group 1:

     P | Q | I2 | __ktk_cube_labels_seed                               | __ktk_cube_labels_enrich
    ===+===+====+==================================================+==================================================
     0 | 1 |  1 | P=0/Q=1/<uuid2>.parquet, P=0/Q=1/<uuid3>.parquet | P=0/Q=1/<uuid7>.parquet
     1 | 0 |  1 | P=1/Q=0/<uuid4>.parquet                          | P=1/Q=0/<uuid8>.parquet, P=0/Q=1/<uuid9>.parquet

    group 2:

     P | Q | I2 | __ktk_cube_labels_seed                               | __ktk_cube_labels_enrich
    ===+===+====+==================================================+==================================================
     1 | 0 |  2 | P=1/Q=0/<uuid5>.parquet                          | P=1/Q=0/<uuid8>.parquet, P=0/Q=1/<uuid9>.parquet

Intra-Partition Joins
`````````````````````
This section explains how DataFrames within a partition within a group are joined.

A simple explanation of the join logic would be: "The coordinates (cube cells) are taken from the seed dataset, all
other information is add via a left join."

Because the user is able to add conditions to the query and because we want to utilize predicate pushdown in a very
efficient way, we define another term: **restricted dataset**. These are datasets which contain
non-:term:`Dimension Column` and non-:term:`Partition Column` to which users wishes to apply restrictions (via
conditions or via partition-by). Because these restrictions always need to apply, we can evaluate them pre-join and
execute an inner join with the seed dataset.

Examples
````````
The following sub-sections illustrate this system in multiple steps.


Example 1 (Join Semantics)
~~~~~~~~~~~~~~~~~~~~~~~~~~
Here, a rather standard example is shown with explanations why data is kept or not::

    columns   = [P, PRED]
    condition = (OK == true) & (SCHED == true)

     Seed    | Conditions                | Enrichments
     db_data | data_checks | schedule    | predictions
    =========+=============+=============+=============
     P=1     | P=1         | P=1         | P=1            <-- included, trivial case
             | OK=true     | SCHED=true  | PRED=0.23
    ---------+-------------+-------------+-------------
     P=2     | P=2         | P=2         | P=2            <-- excluded, because OK=false
             | OK=false    | SCHED=true  | PRED=0.12
    ---------+-------------+-------------+-------------
     P=3     | P=3         | P=3         | P=3            <-- excluded, because SCHED=false
             | OK=true     | SCHED=false | PRED=0.13
    ---------+-------------+-------------+-------------
             | P=4         | P=4         | P=4            <-- excluded, seed is missing
             | OK=true     | SCHED=true  | PRED=0.03          where does this data even come from?!
    ---------+-------------+-------------+-------------
     P=5     | P=5         | P=5         |                <-- included, even though PRED is missing
             | OK=true     | SCHED=true  |
    ---------+-------------+-------------+-------------
     P=6     | P=6         |             | P=6            <-- excluded, SCHED is missing
             | OK=true     |             | PRED=0.01

     ^         ^             ^             ^
     |         |             |             |
     +---------+-------------+             |
               |                           |
           inner join                      |
     tmp1 = db_data <-> data_checks on P   |
     out  = tmp1    <-> schedule    on P   |
     (but order actually doesn't matter)   |
               ^                           |
               |                           |
               +-----------------+---------+
                                 |
                             left join
                                 |
                                 v

                               P | PRED
                              ===+======
                               1 | 0.23
                               5 | NaN


Example 2 (Outer Join)
~~~~~~~~~~~~~~~~~~~~~~
Now, we have a P-L cube, with all datasets except of ``schedule`` having P-L dimensionality::

    columns   = [P, L, PRED]
    condition = (OK == true) & (SCHED == true)

     Seed    | Conditions                | Enrichments
     db_data | data_checks | schedule    | predictions
    =========+=============+=============+=============
     P=1     | P=1         | P=1         | P=1            <-- included, trivial case
     L=1     | L=1         |             | L=1
             | OK=true     | SCHED=true  | PRED=0.23
    ---------+-------------+             +-------------
     P=1     | P=1         |             | P=1            <-- excluded, because OK=false
     L=2     | L=2         |             | L=2
             | OK=false    |             | PRED=0.12
    ---------+-------------+-------------+-------------
     P=2     | P=2         | P=2         | P=2            <-+ excluded, because SCHED=false
     L=1     | L=1         |             | L=1              |
             | OK=true     | SCHED=false | PRED=0.13        |
    ---------+-------------+             +-------------     |
     P=2     | P=2         |             | P=2            <-+
     L=2     | L=2         |             | L=2
             | OK=true     |             | PRED=0.13

     ^         ^             ^             ^
     |         |             |             |
     +---------+-------------+             |
               |                           |
           inner join                      |
     tmp1 = db_data <-> data_checks on P,L |
     out  = tmp1    <-> schedule    on P   |
     (but order actually doesn't matter)   |
               ^                           |
               |                           |
               +-----------------+---------+
                                 |
                             left join
                                 |
                                 v

                             P | L | PRED
                            ===+===+======
                             1 | 1 | 0.23


Example 3 (Projection)
~~~~~~~~~~~~~~~~~~~~~~
This shows how the seed dataset can be used to also produce sub-dimensional / projected results::

    columns   = [P, AVG]
    condition = (SCHED == true)

     Seed    | Conditions  | Enrichments
     db_data | schedule    | agg
    =========+=============+=============
     P=1     | P=1         | P=1            <-- included, trivial case
     L=?     |             |
             | SCHED=true  | AVG=10.2
    ---------+-------------+-------------
     P=2     | P=2         | P=2            <-- excluded, because SCHED=false
     L=?     |             |
             | SCHED=false | AVG=1.34

     ^         ^             ^
     |         |             |
     |         +---+         |
     |             |         |
     project to P  |         |
     |             |         |
     +---------+---+         |
               |             +---------+
           inner join                  |
     out = db_data <-> schedule on P   |
               ^                       |
               |                       |
               +-----------------+-----+
                                 |
                             left join
                                 |
                                 v

                             P |  AVG
                            ===+=======
                             1 |  10.2

Final Concat
~~~~~~~~~~~~
After DataFrames for all partitions in a group are joined, they are concatenated in order of :term:`Partition Columns`.
