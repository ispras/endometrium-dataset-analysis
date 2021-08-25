from setuptools import setup, find_packages

setup(
    name="endometrium-dataset-analysis",
    version="0.1dev",
    packages=find_packages(include=["endoanalysis"]),
    install_requires=[
        "numpy==1.20.2",
        "matplotlib==3.4.1",
    ]
)
