#!/bin/bash
# Script to publish a new version (without PyPI)
# Usage: ./publish-version.sh [patch|minor|major]

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 [patch|minor|major]"
    exit 1
fi

VERSION_TYPE=$1

# Bump version using Poetry
poetry version "$VERSION_TYPE"

# Get the new version
NEW_VERSION=$(poetry version -s)

echo "Version bumped to: $NEW_VERSION"

# Create and push Git tag
git add pyproject.toml
git commit -m "Bump version to $NEW_VERSION" || true
git tag "v$NEW_VERSION"
git push origin main
git push origin "v$NEW_VERSION"

echo "✅ Version $NEW_VERSION published!"
echo "Install with: uv pip install git+https://github.com/atroo/llm-helpers.git@v$NEW_VERSION"

