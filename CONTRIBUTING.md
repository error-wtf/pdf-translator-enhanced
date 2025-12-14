# Contributing to PDF Translator Enhanced

Thank you for your interest in contributing! This document provides guidelines
for contributing to PDF Translator Enhanced.

## 📜 License Agreement

By contributing to this project, you agree that your contributions will be
licensed under the **Anti-Capitalist Software License v1.4**.

This means your code:
- ✅ Can be used for personal, academic, and non-profit purposes
- ✅ Can be used by worker-owned cooperatives
- ❌ Cannot be used by for-profit corporations

## 🛠️ Development Setup

### Prerequisites

- Python 3.10 or higher
- Git
- Ollama (for local testing)

### Setup

```bash
# Clone the repository
git clone https://github.com/error-wtf/pdf-translator-enhanced.git
cd pdf-translator-enhanced

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or: venv\Scripts\activate  # Windows

# Install dependencies with dev extras
pip install -e ".[dev]"

# Run tests to verify setup
pytest tests/ -v
```

## 📝 Code Style

### Formatting

We use **ruff** for linting and formatting:

```bash
# Check code style
ruff check .

# Format code
ruff format .
```

### Guidelines

1. **Line length**: 100 characters maximum
2. **Quotes**: Double quotes for strings
3. **Imports**: Sorted with isort (handled by ruff)
4. **Type hints**: Encouraged but not required
5. **Docstrings**: Required for public functions

### Example

```python
"""
Module description here.

© 2025 Sven Kalinowski with small help of Lino Casu
Licensed under the Anti-Capitalist Software License v1.4
"""
from __future__ import annotations

import logging
from typing import Optional, List

logger = logging.getLogger(__name__)


def translate_text(
    text: str,
    target_language: str,
    model: str = "qwen2.5:7b",
) -> str:
    """
    Translate text to target language.
    
    Args:
        text: Text to translate
        target_language: Target language name
        model: Ollama model to use
    
    Returns:
        Translated text
    """
    # Implementation here
    pass
```

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_core.py -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html
```

### Writing Tests

1. Place tests in the `tests/` directory
2. Name test files `test_*.py`
3. Name test functions `test_*`
4. Use pytest fixtures for setup

```python
import pytest
from formula_isolator import extract_formulas


class TestFormulaIsolator:
    """Tests for formula_isolator module."""
    
    def test_extract_inline_math(self):
        """Test extraction of inline math."""
        text = "The equation $E=mc^2$ is famous."
        result = extract_formulas(text)
        
        assert len(result.formulas) == 1
        assert "$E=mc^2$" in result.formulas[0].original
```

## 🔄 Pull Request Process

### Before Submitting

1. **Create a branch** from `main`:
   ```bash
   git checkout -b feat/your-feature-name
   ```

2. **Make your changes** following the code style guidelines

3. **Add tests** for new functionality

4. **Run tests** to ensure nothing is broken:
   ```bash
   pytest tests/ -v
   ```

5. **Update documentation** if needed

### Submitting

1. Push your branch:
   ```bash
   git push origin feat/your-feature-name
   ```

2. Create a Pull Request on GitHub

3. Fill out the PR template with:
   - Description of changes
   - Related issues
   - Testing done

### PR Review

- All PRs require at least one review
- CI must pass (tests + linting)
- Changes should be rebased on latest `main`

## 📁 Project Structure

```
pdf-translator-enhanced/
├── *.py                 # Core modules
├── tests/               # Unit tests
│   └── test_*.py
├── .github/
│   └── workflows/       # CI/CD
│       └── ci.yml
├── Dockerfile           # Container build
├── docker-compose.yml   # Deployment
├── requirements.txt     # Dependencies
├── pyproject.toml       # Project config
├── README.md            # Documentation
├── CHANGELOG.md         # Release notes
└── CONTRIBUTING.md      # This file
```

## 🎯 Areas for Contribution

### High Priority

- [ ] Additional language support
- [ ] Performance optimizations
- [ ] Better error messages
- [ ] Integration tests

### Medium Priority

- [ ] Documentation improvements
- [ ] UI/UX enhancements
- [ ] New translation backends
- [ ] Caching improvements

### Low Priority

- [ ] Code refactoring
- [ ] Additional CLI commands
- [ ] Example notebooks

## 📬 Getting Help

- **Issues**: Open a GitHub issue for bugs or feature requests
- **Discussions**: Use GitHub Discussions for questions
- **Email**: mail@error.wtf

## 🙏 Acknowledgments

Contributors will be acknowledged in:
- The CHANGELOG for their specific contributions
- The README acknowledgments section

---

© 2025 Sven Kalinowski with small help of Lino Casu  
Licensed under the Anti-Capitalist Software License v1.4
