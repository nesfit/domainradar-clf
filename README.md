# ML Classification Pipeline for DomainRadar

## Requirements
- Python 3.11
- poetry 1.8 (Please install using pipx)

Note: If you have a newer default Python version, make sure this venv uses 3.11, e.g.:
`poetry env use /usr/local/bin/python3.11`

## Usage

Modify the **pyproject.toml** file in your Python Poetry project - extend the `[tool.poetry.dependencies]` section with:
`classifiers = { git = git@github.com:nesfit/domainradar-clf.git, branch = "main }`

Then type:
`poetry update`

In your code, you can use the module like:
```
import pandas as pd
from classifiers import pipeline

# Initialize the classification pipeline
p = Pipeline()

# Load or prepare a dataframe with feature vectors
df = pd.DataFrame({
    "domain_name": ['first.com', 'second.net', ...],
    "lex_name_len": [9, 10, ...],
    ...
})

# Classify!
results = p.classify_domains(df)
print(results)
```

See `example.py` for a working example.

**NOTE: Don't forget** to run you code from `poetry shell`.

## Development

Install Python Poetry.
In the domainradar-clf root directory, enter:
```
poetry install
```

To experiment with the included notebooks, you have to explicitly include
the development dependencies:
```
poetry install --with dev
```

You can run scripts from the **classifiers** directory from
```
poetry shell
```
