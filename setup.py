# setup.py
from pathlib import Path
from setuptools import find_packages, setup

# Load packages from requirements.txt
BASE_DIR = Path(__file__).parent
with open(Path(BASE_DIR, "requirements.txt"), "r") as file:
    required_packages = [ln.strip() for ln in file.readlines()]

# setup.py
setup(
    name="compressor-classifier",
    version=0.1,
    description="Compression text classifier",
    author="Marcelo Destefani",
    author_email="marcelo@hastapronto.cl",
    url="https://github.com/destefani/compressor-classifier",
    python_requires=">=3.7",
    install_requires=[required_packages],
    packages=find_packages(include=['compressor*', 'config*']),

)
