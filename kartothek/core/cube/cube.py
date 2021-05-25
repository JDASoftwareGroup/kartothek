import typing

import attr

from kartothek.core.cube.constants import KTK_CUBE_UUID_SEPERATOR
from kartothek.core.dataset import _validate_uuid
from kartothek.utils.converters import (
    converter_str,
    converter_str_set,
    converter_str_tupleset,
)

__all__ = ("Cube",)


def _validate_not_subset(of, allow_none=False):
    """
    Create validator to check if an attribute is not a subset of ``of``.

    Parameters
    ----------
    of: str
        Attribute name that the subject under validation should not be a subset of.

    Returns
    -------
    validator: Callable
        Validator that can be used for ``attr.ib``.
    """

    def _v(instance, attribute, value):
        if allow_none and value is None:
            return
        other_set = set(getattr(instance, of))
        if isinstance(value, str):
            my_set = {value}
        else:
            my_set = set(value)
        share = my_set & other_set

        if share:
            raise ValueError(
                "{attribute} cannot share columns with {of}, but share the following: {share}".format(
                    attribute=attribute.name, of=of, share=", ".join(sorted(share))
                )
            )

    return _v


def _validate_subset(of, allow_none=False):
    """
    Create validator to check that an attribute is a subset of ``of``.

    Parameters
    ----------
    of: str
        Attribute name that the subject under validation should be a subset of.

    Returns
    -------
    validator: Callable
        Validator that can be used for ``attr.ib``.
    """

    def _v(instance, attribute, value):
        if allow_none and value is None:
            return
        other_set = set(getattr(instance, of))
        if isinstance(value, str):
            my_set = {value}
        else:
            my_set = set(value)
        too_much = my_set - other_set

        if too_much:
            raise ValueError(
                "{attribute} must be a subset of {of}, but it has additional values: {too_much}".format(
                    attribute=attribute.name,
                    of=of,
                    too_much=", ".join(sorted(too_much)),
                )
            )

    return _v


def _validator_uuid(instance, attribute, value):
    """
    Attr validator to validate if UUIDs are valid.
    """
    _validator_uuid_freestanding(attribute.name, value)


def _validator_uuid_freestanding(name, value):
    """
    Freestanding version of :meth:`_validate_not_subset`.
    """
    if not _validate_uuid(value):
        raise ValueError(
            '{name} ("{value}") is not compatible with kartothek'.format(
                name=name, value=value
            )
        )
    if value.find(KTK_CUBE_UUID_SEPERATOR) != -1:
        raise ValueError(
            '{name} ("{value}") must not contain UUID separator {sep}'.format(
                name=name, value=value, sep=KTK_CUBE_UUID_SEPERATOR
            )
        )


def _validator_not_empty(instance, attribute, value):
    """
    Attr validator to validate that a list is not empty:
    """
    if len(value) == 0:
        raise ValueError("{name} must not be empty".format(name=attribute.name))


@attr.s(frozen=True)
class Cube:
    """
    OLAP-like cube that fuses multiple datasets.

    Parameters
    ----------
    dimension_columns: Tuple[str, ...]
        Columns that span dimensions. This will imply index columns for the seed dataset, unless
        the automatic index creation is suppressed via ``suppress_index_on``.
    partition_columns: Tuple[str, ...]
        Columns that are used to partition the data. They also create (implicit) primary indices.
    uuid_prefix: str
        All datasets that are part of the cube will have UUIDs of form ``'uuid_prefix++ktk_cube_dataset_id'``.
    seed_dataset: str
        Dataset that present the ground-truth regarding cells present in the cube.
    index_columns: Tuple[str, ...]
        Columns for which secondary indices will be created. They may also be part of non-seed datasets.
    suppress_index_on: Tuple[str, ...]
        Suppress auto-creation of an index on the given dimension columns. Must be a subset of ``dimension_columns``
        (other columns are not subject to automatic index creation).
    """

    dimension_columns = attr.ib(
        converter=converter_str_tupleset,
        type=typing.Tuple[str, ...],
        validator=[_validator_not_empty],
    )

    partition_columns = attr.ib(
        converter=converter_str_tupleset,
        type=typing.Tuple[str, ...],
        validator=[_validator_not_empty, _validate_not_subset("dimension_columns")],
    )

    uuid_prefix = attr.ib(
        converter=converter_str, type=str, validator=[_validator_uuid]
    )

    seed_dataset = attr.ib(
        converter=converter_str, default="seed", type=str, validator=[_validator_uuid]
    )

    index_columns = attr.ib(
        converter=converter_str_set,
        default=None,
        type=typing.FrozenSet[str],
        validator=[
            _validate_not_subset("dimension_columns"),
            _validate_not_subset("partition_columns"),
        ],
    )

    suppress_index_on = attr.ib(
        converter=converter_str_set,
        default=None,
        type=typing.FrozenSet[str],
        validator=[_validate_subset("dimension_columns", allow_none=True)],
    )

    def ktk_dataset_uuid(self, ktk_cube_dataset_id):
        """
        Get Kartothek dataset UUID for given dataset UUID, so the prefix is included.

        Parameters
        ----------
        ktk_cube_dataset_id: str
            Dataset ID w/o prefix

        Returns
        -------
        ktk_dataset_uuid: str
            Prefixed dataset UUID for Kartothek.

        Raises
        ------
        ValueError
            If ``ktk_cube_dataset_id`` is not a string or if it is not a valid UUID.
        """
        ktk_cube_dataset_id = converter_str(ktk_cube_dataset_id)
        _validator_uuid_freestanding("ktk_cube_dataset_id", ktk_cube_dataset_id)
        return "{uuid_prefix}{sep}{ktk_cube_dataset_id}".format(
            uuid_prefix=self.uuid_prefix,
            sep=KTK_CUBE_UUID_SEPERATOR,
            ktk_cube_dataset_id=ktk_cube_dataset_id,
        )

    @property
    def ktk_index_columns(self):
        """
        Set of all available index columns through Kartothek, primary and secondary.
        """
        # FIXME: do not always add dimension columns. Also, check all users of this property!
        return (
            set(self.partition_columns)
            | set(self.index_columns)
            | (set(self.dimension_columns) - set(self.suppress_index_on))
        )

    def copy(self, **kwargs):
        """
        Create a new cube specification w/ changed attributes.

        This will not trigger any IO operation, but only affects the cube specification.

        Parameters
        ----------
        kwargs: Dict[str, Any]
            Attributes that should be changed.

        Returns
        -------
        cube: Cube
            New abstract cube.
        """
        return attr.evolve(self, **kwargs)
