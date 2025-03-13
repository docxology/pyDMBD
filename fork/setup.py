from setuptools import setup, find_packages

setup(
    name="fork",
    version="0.1.0",
    description="Documentation and examples for Dynamic Markov Blanket Detection (DMBD)",
    author="DMBD Team",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.19.0",
        "matplotlib>=3.3.0",
        "scipy>=1.7.0",
        "pandas>=1.3.0",
        "pytest>=6.2.0",
        "pytest-cov>=2.12.0",
        "jupyter>=1.0.0",
        "ipykernel>=6.0.0",
        "nbformat>=5.1.0",
        "nbconvert>=6.0.0",
        "sphinx>=4.0.0",
        "sphinx-rtd-theme>=0.5.0",
        "myst-parser>=0.15.0",
    ],
    extras_require={
        "dev": [
            "black",
            "flake8",
            "isort",
            "mypy",
            "pre-commit",
        ],
        "test": [
            "pytest",
            "pytest-cov",
            "pytest-xdist",
        ],
        "docs": [
            "sphinx",
            "sphinx-rtd-theme",
            "myst-parser",
        ],
    }
) 