from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ecg-llm-diagnosis",
    version="0.1.0",
    author="Aayush Parashar",
    author_email="aayush.parashar@example.com",
    description="Large Language Model-Powered ECG Analysis for Cardiovascular Diagnosis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Anjaniputra15/ECG-LLM-Medical-research",
    project_urls={
        "Bug Tracker": "https://github.com/Anjaniputra15/ECG-LLM-Medical-research/issues",
        "Documentation": "https://github.com/Anjaniputra15/ECG-LLM-Medical-research/docs",
        "Source Code": "https://github.com/Anjaniputra15/ECG-LLM-Medical-research",
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pre-commit>=3.0.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "myst-parser>=0.18.0",
        ],
        "cuda": [
            "torch-audio==2.0.1",
            "torch-vision==0.15.1",
        ],
        "research": [
            "wandb>=0.15.0",
            "mlflow>=2.0.0",
            "optuna>=3.0.0",
            "tensorboard>=2.10.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ecg-llm=src.cli:main",
            "ecg-preprocess=scripts.preprocess_ecg:main",
            "ecg-train=scripts.train_model:main",
            "ecg-evaluate=scripts.evaluate:main",
        ],
    },
    include_package_data=True,
    package_data={
        "src": ["config/*.yaml", "data/*.json"],
    },
    keywords=[
        "ecg", "electrocardiogram", "llm", "large language model", 
        "medical ai", "cardiovascular", "diagnosis", "healthcare", 
        "machine learning", "deep learning", "signal processing"
    ],
    zip_safe=False,
)