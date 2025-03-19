from setuptools import setup, find_packages
import os

# Read the contents of README.md if it exists
readme_path = os.path.join(os.path.dirname(__file__), "README.md")
if os.path.exists(readme_path):
    with open(readme_path, "r", encoding="utf-8") as fh:
        long_description = fh.read()
else:
    long_description = "Quantum-MetaLearn: A project for meta-learning experiments."

setup(
    name="quantum-metalearn",
    version="1.1.1",
    author="Krishna Bajpai",
    author_email="bajpaikrishna715@gmail.com",
    description="A project for meta-learning experiments",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    packages=find_packages(include=["metalearn*"]),
    install_requires=[
        "torch>=2.0.0",
        "gymnasium>=0.28.0",
        "numpy>=1.23.0",
        "tqdm>=4.65.0",
        "click>=8.1.0"
    ],
    entry_points={
        'console_scripts': [
            'metalearn=metalearn.interfaces.cli:cli'
        ]
    },
    python_requires=">=3.9",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
)
