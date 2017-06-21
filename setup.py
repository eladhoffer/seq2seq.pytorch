# -*- coding: utf-8 -*-
from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='seq2seq',
    version='0.1.0',
    description='Sequence-to-Sequnce learning in PyTorch',
    long_description=readme,
    author='Elad Hoffer',
    author_email='elad.hoffer@gmail.com',
    url='https://github.com/eladhoffer/seq2seq.pytorch',
    license=license,
    packages=find_packages(exclude=('examples', 'results', 'scripts', 'tests'))
)
