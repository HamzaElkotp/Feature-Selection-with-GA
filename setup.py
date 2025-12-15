from setuptools import setup, find_packages

setup(
    name="Feature Extraction with GA",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "pygad",
        "ttkbootstrap",
        "ttkbootstrap-icons",
        "pydantic",
        "pathlib",
        "scikit-learn",  # Explicitly include sklearn
    ],
    extras_require={
        "gpu": [
            "cupy-cuda11x>=12.0.0",  # GPU acceleration for CUDA 11.x (RTX 3050 compatible)
        ],
        "gpu-cuda12": [
            "cupy-cuda12x>=12.0.0",  # GPU acceleration for CUDA 12.x
        ],
    },
    entry_points={
        "console_scripts": [
            "start-app = main:main",  # allows running "start-app" from terminal
        ]
    },
    include_package_data=True,
)