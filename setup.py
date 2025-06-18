from setuptools import setup, find_packages
import os

# Read requirements
def read_requirements(filename):
    with open(filename) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

# Get the directory containing this file
here = os.path.abspath(os.path.dirname(__file__))

# Read requirements from requirements.txt
requirements = read_requirements(os.path.join(here, 'requirements.txt'))

setup(
    name='gym_nim',
    version='0.2.0',
    description='A Nim game environment for Gymnasium',
    url="https://github.com/nczempin/gym-nim",
    author='Nicolai Czempin',
    packages=find_packages(),
    install_requires=requirements,
    python_requires='>=3.9',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
)  
