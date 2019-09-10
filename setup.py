from setuptools import setup

with open("requirements.txt") as req_file:
    requirements = req_file.read().splitlines()

setup(
    name="catheter_utils",
    version="0.1.0",
    packages=["catheter_utils"],
    install_requires=requirements,
)
