import os
from setuptools import find_packages
from setuptools import setup


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


requirements = read("requirements.txt").split()


setup(
    name="Deep4Rec",
    version="0",
    author="Marianne Monteiro",
    author_email="mariannelinharesm@gmail.com",
    description=(
        "Popular Deep Learning based recommendation algorithms built on top of TensorFlow served on a simple API"
    ),
    license="Apache-2.0",
    keywords="recommendation recsys deep learning",
    packages=find_packages(exclude=["docs"]),
    include_package_data=True,
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    url="https://github.com/analytics-ufcg/Deep4Rec",
    install_requires=requirements,
    setup_requires=["pytest-runner"],
    tests_require=["pytest", "pytest-flake8"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
