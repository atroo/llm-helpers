# LLM Helpers

A Python library with utility functions for interacting with various LLM providers.

## Installation

### With Poetry (recommended)

```bash
poetry install
```

### For development

```bash
poetry install --with dev
```

### From source (pip)

```bash
pip install -e .
```


### Install the library

- currently only via ssh: `poetry add git+ssh://git@github.com/atroo/llm-helpers.git`

## Usage

### File Upload Conversion

Convert FastAPI uploaded files to the appropriate format for different LLM providers:

```python
from fastapi import FastAPI, UploadFile
from llm_helpers import file_to_message

app = FastAPI()

@app.post("/upload")
async def upload_file(file: UploadFile, provider: str):
    message = await file_to_message(file, provider)
    # Use the message with your LLM provider
    return {"message": message}
```

Supported providers:
- `openai` - OpenAI format
- `azure` - Azure OpenAI format
- `groq` - Not yet implemented

## Development

### Install dependencies

```bash
poetry install --with dev
```

### Running tests

```bash
poetry run pytest
```

### Code formatting

```bash
poetry run black src/ tests/
```

### Linting

```bash
poetry run ruff check src/ tests/
```

### Type checking

```bash
poetry run mypy src/
```

### Add a new dependency

```bash
poetry add package-name
```

## Project Structure

```
llm-helpers/
├── src/
│   └── llm_helpers/
│       ├── __init__.py
│       └── file_utils.py
├── tests/
│   ├── __init__.py
│   └── test_file_utils.py
├── pyproject.toml
├── README.md
└── .gitignore
```

## License

MIT