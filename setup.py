from setuptools import setup, find_packages

setup(
    name="dl-pneumonia",
    version="0.1.0",
    packages=find_packages(),
    description="Deep Learning project for pneumonia detection",
    author="Your Name",
    python_requires=">=3.6",
    install_requires=[
        "numpy",
        "Pillow", 
        "albumentations",
        "tqdm",

    ]
) 