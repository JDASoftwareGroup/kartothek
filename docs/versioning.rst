.. _versioning:

Versioning
==========

Specification
-------------

The storage specification documented in :ref:`dataset_spec` is versioned using
the integer dataset attribute `metadata_version` which can be defined for all
writing pipelines. All `metadata_version` changes will be reflected in the
version by increasing it.
New specification versions will be introduced by increasing the *minor* library
version while a removal of a supported version (both read and write) will
trigger an increase of the *major* version number.


Library API
-----------

Kartothek exposes a dedicated module to highlight which functions, classes, etc. are considered stable API and are publicly available. We intend to apply [semantic
versioning](https://semver.org) to the best of our ability for the API exposed in :mod:`kartothek.api`. See also :doc:`api`.

We are aware that the API not only consists of the signatures of functions and
types of objects exposed by a library but as well by its behaviour. Many changes
involve judgement calls about the compatibility of a given change and it might
be possible that we'll introduce behavioural changes which you would not
perceive compatible with semantic versioning. We will adhere as best as possible
to this convention. In any case we intend to document all intended changes,
whether breaking or not, in our changelog, see :doc:`changes`.
