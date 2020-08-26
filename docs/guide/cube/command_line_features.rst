
Command Line Features
---------------------

.. raw:: html

       <style>
          .term-fg1 {
             font-weight: bold;
          }
          .term-fg4 {
             text-decoration: underline;
          }
          .term-fg33 {
             color: orange;
          }
       </style>

Kartothek Cube also features a command line interface (CLI) for some cube operations. To use it, create a ``skv.yml`` file that
describes `storefact`_ stores:

.. code-block:: yaml

   dataset:
      type: hfs
      path: path/to/data

Now use the ``kartothek_cube`` command to gather certain cube information:

.. code-block:: bash

   kartothek_cube geodata info

.. raw:: html

   <pre>
   <span class="term-fg33 term-fg1 term-fg4">Infos</span>
   <span class="term-fg1">UUID Prefix:</span>        geodata
   <span class="term-fg1">Dimension Columns:</span>
     - city: string
     - day: timestamp[ns]
   <span class="term-fg1">Partition Columns:</span>
     - country: string
   <span class="term-fg1">Index Columns:</span>
   &nbsp;
   <span class="term-fg1">Seed Dataset:</span>      seed
   &nbsp;
   <span class="term-fg33 term-fg1 term-fg4">Dataset: latlong</span>
   <span class="term-fg1">Partition Keys:</span>
     - country: string
   <span class="term-fg1">Partitions:</span> 4
   <span class="term-fg1">Metadata:</span>
     {
       &quot;creation_time&quot;: &quot;2019-10-01T12:11:38.263496&quot;,
       &quot;klee_dimension_columns&quot;: [
         &quot;city&quot;,
         &quot;day&quot;
       ],
       &quot;klee_is_seed&quot;: false,
       &quot;klee_partition_columns&quot;: [
         &quot;country&quot;
       ]
     }
   <span class="term-fg1">Dimension Columns:</span>
     - city: string
   <span class="term-fg1">Payload Columns:</span>
     - latitude: double
     - longitude: double
   &nbsp;
   <span class="term-fg33 term-fg1 term-fg4">Dataset: seed</span>
   <span class="term-fg1">Partition Keys:</span>
     - country: string
   <span class="term-fg1">Partitions:</span> 3
   <span class="term-fg1">Metadata:</span>
     {
       &quot;creation_time&quot;: &quot;2019-10-01T12:11:38.206653&quot;,
       &quot;klee_dimension_columns&quot;: [
         &quot;city&quot;,
         &quot;day&quot;
       ],
       &quot;klee_is_seed&quot;: true,
       &quot;klee_partition_columns&quot;: [
         &quot;country&quot;
       ]
     }
   <span class="term-fg1">Dimension Columns:</span>
     - city: string
     - day: timestamp[ns]
   <span class="term-fg1">Payload Columns:</span>
     - avg_temp: int64
   &nbsp;
   <span class="term-fg33 term-fg1 term-fg4">Dataset: time</span>
   <span class="term-fg1">Partitions:</span> 1
   <span class="term-fg1">Metadata:</span>
     {
       &quot;creation_time&quot;: &quot;2019-10-01T12:11:41.734913&quot;,
       &quot;klee_dimension_columns&quot;: [
         &quot;city&quot;,
         &quot;day&quot;
       ],
       &quot;klee_is_seed&quot;: false,
       &quot;klee_partition_columns&quot;: [
         &quot;country&quot;
       ]
     }
   <span class="term-fg1">Dimension Columns:</span>
     - day: timestamp[ns]
   <span class="term-fg1">Payload Columns:</span>
     - month: int64
     - weekday: int64
     - year: int64
   </pre>

Some information is not available when reading the schema information and require a cube scan:

.. code-block:: bash

   kartothek_cube geodata stats

.. raw:: html

   <pre>
   [########################################] | 100% Completed |  0.1s
   <span class="term-fg33 term-fg1 term-fg4">latlong</span>
   <span class="term-fg1">blobsize:</span>  5,690
   <span class="term-fg1">files:</span>  4
   <span class="term-fg1">partitions:</span>  4
   <span class="term-fg1">rows:</span>  4
   &nbsp;
   <span class="term-fg33 term-fg1 term-fg4">seed</span>
   <span class="term-fg1">blobsize:</span>  4,589
   <span class="term-fg1">files:</span>  3
   <span class="term-fg1">partitions:</span>  3
   <span class="term-fg1">rows:</span>  8
   &nbsp;
   <span class="term-fg33 term-fg1 term-fg4">time</span>
   <span class="term-fg1">blobsize:</span>  3,958
   <span class="term-fg1">files:</span>  1
   <span class="term-fg1">partitions:</span>  1
   <span class="term-fg1">rows:</span>  366
   &nbsp;
   <span class="term-fg33 term-fg1 term-fg4">__total__</span>
   <span class="term-fg1">blobsize:</span>  14,237
   <span class="term-fg1">files:</span>  8
   </pre>


Use ``kartothek_cube --help`` to get a list of all commands, or see :mod:`kartothek_cube.cli`.

.. _storefact: https://github.com/blue-yonder/storefact

