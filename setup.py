from os import path
from setuptools import setup, find_packages


here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setup(name='mae_lasso',
      version='0.0.0',
      description='L1 regression with lasso penalty',
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='https://github.com/steppi/mae_lasso'
      author='Albert Steppi'
      author_email='albert.steppi@gmail.com',
      classifiers=[
          'Development Status :: 4 - Beta',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7'
      ],
      packages=find_packages(),
      install_requires=['scikit-learn>=0.20.0', 'numpy']
      extras_require={'test': ['nose', 'coverage', 'python-coveralls'],
                      'cplex'}
      )
