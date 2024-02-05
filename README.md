# ML Classification Pipeline for DomainRadar

## Usage

Modify the **pyproject.toml** file in your Python Poetry project - extend the `[tool.poetry.dependencies]` section with:
`classifiers = { git = git@github.com:ihranicky/domainradar-clf.git, branch = "main }`

Then type:
`poetry update`

In your code, you can use the module like:
```
from classifiers import pipeline

p = Pipeline()
result = p.classifyDomain('mydomain.com', {'dns_A_count': 2, 'geo_mean_lat': 35.6980, ...})
print(result)
```

Don't forget to run you code from `poetry shell`.

## Development

Install Python Poetry.
In the domainradar-clf root directory, enter:
```
poetry install
```
You can run scripts from the **classifiers** directory from
```
poetry shell
```