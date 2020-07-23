.. _versioning:

Versioning
==========

Specification
-------------

The storage specification documented in :ref:`dataset_spec` is versioned using
the integer dataset attribute `metadata_version` which can be defined for all
writing pipelines. All `metadata_version` changes will be reflected in the
version by increasing in.
New specification versions will be introduced by increasing the *minor* library
version while a removal of a supported version (both read and write) will
trigger an increase of the *major* version number.


Library API
-----------
The library versioning itself uses a slighlty adapted version of [semantic
versioning](https://semver.org) since the various modules of `kartothek` reached
different levels of maturity.

We commit to full semantic versioning for the subpackages

* :mod:`kartothek.serialization`
* :mod:`kartothek.io`
* :mod:`kartothek.api`

which we consider as our stable public API and should be the primary entrypoint
for user interactions.

The subpackages

* :mod:`kartothek.core`
* :mod:`kartothek.io_components`
* :mod:`kartothek.utils`

should be used primarily by extension developers and not the ordinary end user.
We allow ourselves therefore a bit more freedom and permit breaking interface
changes in every *minor* version. For changes in :mod:`~kartothek.core`, :mod:`kartothek.utils`
and :mod:`~kartothek.io_components` with expected impact on end users we will decide
on an individual case basis if an increase of major version is justified to
signal the user potential breakage.
Under any circumstances, breaking interface changes will be documented to the
best of our knowledge.
