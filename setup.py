"""setup.py — makes lung_seg installable as a local package."""
from setuptools import setup, find_packages

setup(
    name="lung_seg",
    version="1.0.0",
    packages=find_packages(exclude=["tests*", "notebooks*"]),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.38.0",
        "albumentations>=1.4.0",
        "scipy>=1.12.0",
        "scikit-image>=0.22.0",
        "omegaconf>=2.3.0",
        "pandas>=2.2.0",
        "matplotlib>=3.8.0",
        "tqdm>=4.66.0",
        "Pillow>=10.0.0",
    ],
)
