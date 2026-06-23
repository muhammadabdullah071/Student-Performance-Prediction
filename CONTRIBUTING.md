# Contributing to Student Performance Prediction

We love contributions! Here's how you can help make this project better.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/<your-username>/Student-Performance-Prediction.git`
3. Create a virtual environment: `python -m venv venv`
4. Install dependencies: `pip install -r requirements.txt`
5. Create a branch: `git checkout -b feature/your-feature-name`

## Development Guidelines

### Code Quality
- All Python code must pass `flake8`, `black`, and `isort` checks
- Type hints required for all function signatures
- Docstrings required for all public functions and classes
- Maximum line length: 100 characters

### Running Tests
```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=term-missing
```

### Code Style
```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Lint
flake8 src/ tests/

# Type check
mypy src/
```

## Pull Request Process

1. Update `README.md` with details of changes if needed
2. Update the Jupyter notebook if you change the analysis
3. Update `requirements.txt` if you add dependencies
4. Ensure all tests pass and coverage doesn't decrease
5. Your PR must be reviewed by at least one maintainer

## Dataset Contributions

If you have access to additional student performance data:
- Add raw CSV files to `data/raw/`
- Document the source and schema in the notebook
- Update the preprocessing pipeline to handle the new format

## Report Issues

Found a bug? Open an issue with:
- A clear title and description
- Steps to reproduce
- Expected vs actual behavior
- Screenshots if applicable

## Feature Requests

Suggest features by opening an issue with the "enhancement" tag. We welcome:
- New ML models
- Feature engineering techniques
- Deployment improvements
- UI/UX suggestions

## Code of Conduct

Be respectful, inclusive, and constructive. We're all here to learn and grow together.

Thank you for contributing! 🚀