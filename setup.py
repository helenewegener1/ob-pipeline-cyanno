from setuptools import setup, find_packages

setup(
    name="cyannotool",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn"
    ],
    entry_points={
        "console_scripts": [
            "cyannotool=module.entrypoint_cyanno:main"
        ]
    },
)
