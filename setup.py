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
        "pathlib"
    ],
    entry_points={
        "console_scripts": [
            "start-app = main:main",  # allows running "start-app" from terminal
        ]
    },
    include_package_data=True,
)