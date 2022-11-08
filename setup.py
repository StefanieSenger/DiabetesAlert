from setuptools import find_packages
from setuptools import setup

with open('requirements.txt') as f:
    content = f.readlines()
requirements = [x.strip() for x in content]

setup(name='ml-zoomcamp_midterm-project',
      version="0.0.1",
      description="train and predict",
      install_requires=requirements,
      packages=find_packages())
