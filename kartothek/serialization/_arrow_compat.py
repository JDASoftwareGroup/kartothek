import json


def _fix_pyarrow_07992_table(table):
    # Parquet files written by pyarrow 0.7.992 have Pandas metadata that is
    # rejected by pyarorw 0.8.0, fix this.
    if table.schema.metadata is not None and b"pandas" in table.schema.metadata:
        pandas_metadata = json.loads(table.schema.metadata[b"pandas"].decode("utf8"))
        new_columns = []
        has_changed = False
        column_names = set()
        for col in pandas_metadata["columns"]:
            if col["name"] is None:
                col["name"] = "__index_level_0__"
                has_changed = True
            new_columns.append(col)
            column_names.add(col["name"])

        if (pandas_metadata["index_columns"] == ["__index_level_0__"]) and not (
            "__index_level_0__" in column_names
        ):
            # really old files are missing index information
            has_changed = True
            pandas_type = "int64"
            new_columns.append(
                {
                    "metadata": None,
                    "name": "__index_level_0__",
                    "numpy_type": pandas_type,
                    "pandas_type": pandas_type,
                }
            )

        pandas_metadata["columns"] = new_columns
        if has_changed:
            meta = json.dumps(pandas_metadata).encode("utf-8")
            table = table.replace_schema_metadata({b"pandas": meta})
    return table
