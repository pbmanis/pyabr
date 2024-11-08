from setuptools import setup, find_packages
import setuptools
import os, shutil


import os

path = os.path.join(os.path.dirname(__file__), 'src/')
version = None
for line in open(os.path.join(path, '__init__.py'), 'r').readlines():
    if line.startswith('__version__'):
        version = line.partition('=')[2].strip('"\' \n')
        break
if version is None:
    raise Exception("Could not read __version__ from src/__init__.py")

setup(name='pyABR',
      version=version,
      description='ABR Acquisition with Python',
      url='http://github.com/pbmanis/pyabr3',
      author='Paul B. Manis, Tessa J.F. Ropp',
      author_email='pmanis@med.unc.edu',
      license='MIT',
      packages=find_packages(include=['src*']),
      entry_points={
          'console_scripts': [
               'testsound=tests.play_test_sounds:main',
               "SC=pysound.stimController:main",
               ],

      },
      zip_safe=False,
      # data_files=[('wav', ['*.wav']),
      #             ('p;, [*.p]')],  # includes the current compiled mechanisms
#      cmdclass={'makeneuron': 'Build_Nmodl'},
      classifiers = [
             "Programming Language :: Python :: 3.10+",
             "Development Status ::  Beta",
             "Environment :: Console",
             "Intended Audience :: Neuroscientists, computational",
             "License :: MIT",
             "Operating System :: OS Independent",
             "Topic :: Software Development :: Tools :: Python Modules",
             ],
      )
      