## How to contribute

### Bugs

If you've found a bug, please report it to the issue tracker and

* describe the bug you encountered and what the expected behavior should be,
* provide a [mcve](https://stackoverflow.com/help/mcve) if possible, and
* be as explicit about your environment as possible, e.g. provide a `pip freeze` / `conda list`.

### Development

#### Setup

##### Installation using conda

To get started with conda, install conda according to the official guide, set up a conda environment and install all
requirements.

```bash
conda create --name conda_kartothek python=3.7
conda activate conda_kartothek
conda install pip
pip install -e .
pip install -r test-requirements.txt
```

##### Installation using pip

---
###### Note:

You will have to use conda in order to be able to conveniently build and test the documentation. The docs component does
not support pip.

---

To get started with pip, set up a new virtual environment and install all requirements.

```bash
virtualenv kartothek-dev --python=python3.7
source kartothek-dev/bin/activate
pip install -e .
pip install -r test-requirements.txt
```

#### Running tests

We're using [pytest](https://pytest.org) as a testing framework and make heavy use of
`fixtures` and `parametrization`.

To run the tests simply run

```bash
pytest
```

#### Running benchmarks

For performance critical code paths we have [asv](https://pre-commit.com) benchmarks in place in
the subfolder `asv_bench`.
To run the benchmarks a single time and receive immediate feedback run

```bash
asv run --python=same --show-stderr
```

#### Building documentation

##### Docs setup

---
###### Note:

Kartothek does not supply a pip requirements provisions for docs. Please use conda for the best setup experience.

---

Build the `kartothek-docs` conda environment.

```bash
conda env create -f docs/environment-docs.yml
```

##### Build documentation

Activate the `kartothek-docs` conda environment and build the documentation.

```bash
conda activate kartothek-docs
python setup.py docs
```

#### Code style

To ensure a consistent code style across the code base we're using `black`, `flake8`,
and `isort` for formatting and linting.

We have pre-commit hooks for all of these tools which take care of formatting
and checking the code. To set up these hooks, please follow the installation
instructions on the [pre-commit](https://pre-commit.com) homepage.

If you prefer to perform manual formatting and linting, you can run the necessary
toolchain like this

```bash
./format_code.sh
```
