from __future__ import absolute_import,print_function
from setuptools import setup,find_packages
import os

setup(name='KGDiff',
      version=0.1,
      description='KGDiff packages',
      author='Zhiguang Fan',
      license='MIT',
      packages=find_packages(),

      zip_safe=False,
      include_package_data=True
      )