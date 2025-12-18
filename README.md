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

- `poetry add http://git@github.com/atroo/llm-helpers.git`

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

```bash
poetry run pytest -k test_get_llm_google
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

## Publishing a New Version

### 1. Bump the version

You can use Poetry to automatically bump the version:

```bash
# Patch version (0.1.1 → 0.1.2)
poetry version patch

# Minor version (0.1.1 → 0.2.0)
poetry version minor

# Major version (0.1.1 → 1.0.0)
poetry version major
```

Or manually edit the `version` field in `pyproject.toml`.

### 2. Build the package

```bash
poetry build
```

This creates distribution files in the `dist/` directory.

### 3. Publish to PyPI

```bash
# First time: configure PyPI credentials
poetry config pypi-token.pypi your-pypi-token

# Publish (builds automatically if needed)
poetry publish

# Or build and publish separately
poetry build
poetry publish
```

### 4. Install with UV in another project

Once published to PyPI, you can install it in another project using UV:

```bash
# Install specific version
uv pip install llm-helpers==0.1.2

# Install latest version
uv pip install llm-helpers

# Or add to your project's pyproject.toml or requirements.txt
# Then run: uv pip install -r requirements.txt
```

**Alternative: Install directly from Git (without publishing to PyPI)**

If you want to install from Git without publishing to PyPI:

```bash
# Install from a specific tag/version
uv pip install git+https://github.com/atroo/llm-helpers.git@v0.1.2

# Install from a branch
uv pip install git+https://github.com/atroo/llm-helpers.git@main

# Install from a private repository (requires SSH key setup)
uv pip install git+ssh://git@github.com/atroo/llm-helpers.git@v0.1.2
```

**Note:** When installing from Git, make sure to tag your releases:
```bash
git tag v0.1.2
git push origin v0.1.2
```

## License

MIT