from setuptools import setup, find_packages

setup(
    name="endometrium-dataset-analysis",
    version="0.1dev",
    packages=find_packages(include=["endoanalysis"]),
    scripts=[
        "scripts/dab_values",
        "scripts/detect_background",
        "scripts/generate_masks",
        "scripts/resize_dataset",
        "scripts/endonuke_to_coco"
    ],
    install_requires=[
        "tqdm==4.59.0",
        "numpy==1.22.3",
        "matplotlib==3.4.1",
        "seaborn==0.11.1",
        "scipy==1.7.1",
        "scikit-learn==0.24.1",
        "opencv-python==4.5.1.48",
        "pandas==1.2.4",
        "pingouin==0.4.0",
        "pyyaml==5.4.1",
        "scikit-image==0.19.1"
        ]
)
