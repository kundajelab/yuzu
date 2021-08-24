from setuptools import setup

setup(
    name='yuzu-ism',
    version='0.0.1',
    author='Jacob Schreiber',
    author_email='jmschreiber91@gmail.com',
    packages=['yuzu'],
    url='http://pypi.python.org/pypi/yuzu-ism/',
    license='LICENSE.txt',
    description='yuzu implements a compressed-sensing based approach for speeding up saliency calculations.',
    install_requires=[
        "numpy >= 1.14.2",
        "torch >= 1.9.0",
    ],
)