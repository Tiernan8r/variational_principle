from setuptools import setup, find_packages

with open("requirements.txt", "r") as req:
    required=req.read().splitlines()

setup(
    name='variational_principle',
    version='2.0',
    packages=find_packages(
        where="variational_principle/"
    ),
    install_requires=required,
)
