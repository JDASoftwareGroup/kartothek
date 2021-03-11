def test_docs_use_api():
    import os

    files_to_check = []
    for root, _, files in os.walk("../docs"):
        for f in files:
            if f.endswith(".rst"):
                files_to_check.append(os.path.join(root, f))

    for file_ in files_to_check:
        with open(file_) as fd:
            content = fd.read()

        pattern = r"(from|import) kartothek\.(?!(api))"
        import re

        if re.search(pattern, content):
            raise AssertionError(f"Found non-api import in document {file_}")
