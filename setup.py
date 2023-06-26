from setuptools import setup
import os

setup(name='boxpy',
      version='0.0.1',
      author='Nicolas Nytko',
      author_email='nnytko2@illinois.edu',
      packages=['boxpy']
      install_requires=['pyamg', 'scipy', 'numpy']
)
