# -*- coding: utf-8 -*-


from urllib.parse import quote as urllib_quote
from urllib.parse import unquote as urllib_unquote


def quote(value):
    """
    Performs string encoding and urlencodes the given string. If the given value is not
    of string type, it will be cast using the `str` constructor
    """
    if not isinstance(value, (str, bytes)):
        value = str(value)
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    r_val = urllib_quote(value.encode("utf-8"))
    return r_val


def unquote(value):
    """
    Decodes a urlencoded string and performs necessary decoding depending on the used python version.
    """
    return urllib_unquote(value)


def decode_key(key):
    """
    Split a given key into its kartothek components `{dataset_uuid}/{table}/{key_indices}/{filename}`

    Example:
        `uuid/table/index_col=1/index_col=2/partition_label.parquet`

    Returns
    -------
    dataset_uuid: str
    table: str
    key_indices: list
        The already unquoted list of index pairs
    file_: str
        The file name
    """
    key_components = key.split("/")
    dataset_uuid = key_components[0]
    if len(key_components) < 3:
        return key, None, [], None
    table = key_components[1]
    file_ = key_components[-1]
    key_indices = unquote_indices(key_components[2:-1])
    return dataset_uuid, table, key_indices, file_


def quote_indices(indices):
    """
    Urlencode a list of column-value pairs and encode them as:

        `quote(column)=quote(value)`

    Parameters
    ----------
    indices: list of tuple
        A list of tuples where each list entry is (column, value)

    Returns
    -------
    list
        List with urlencoded column=value strings
    """
    quoted_pairs = []
    for column, value in indices:
        quoted_pairs.append(
            "{column}={value}".format(column=quote(column), value=quote(value))
        )

    return quoted_pairs


def unquote_indices(index_strings):
    """
    Take a list of encoded column-value strings and decode them to tuples

    input: `quote(column)=quote(value)`
    output `(column, value)`

    Parameters
    ----------
    indices: list of tuple
        A list of tuples where each list entry is (column, value)

    Returns
    -------
    list
        List with column value pairs
    """
    indices = []
    for index_string in index_strings:
        split_string = index_string.split("=")
        if len(split_string) == 2:
            column, value = split_string
            indices.append((unquote(column), unquote(value)))
    return indices
