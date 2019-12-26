from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()


setup(
    name="bearx",
    version="0.1",
    scripts=['bearx'],
    author="Bartek Szmelczynski",
    author_email="bartek.szmelczynski@gmail.com",
    description=("deep learning library created in order to get in ",
                 "depth knownledge about neural nets"),
    long_description=long_description,
    url="https://github.com/bartekkz/BearX",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ],
)
