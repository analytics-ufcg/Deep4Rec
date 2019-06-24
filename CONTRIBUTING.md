# Contributing

## Install dependencies

```
pip install -r requirements.txt
pip install -r requirements_dev.txt
```

## Setup pre-commit

```
pip install pre-commit
pre-commit install
```

## Running tests

```
pytest -p no:warnings
```

# Generate documentation

## Docstrings

We use [google style docstrings](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html).

## Generated documentation

We use sphinx to automatically generate documentation.

`make html`

