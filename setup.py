from setuptools import setup

with open("requirements.txt", "r") as req:
    required=req.read().splitlines()

setup(
    name='variational_principle',
    version='2.0',
    packages=['variational_principle'],
    install_requires=required,
)