from setuptools import setup, find_packages

setup(
    name='action_lipo',
    version='0.1.0',
    description='A lightweight post-optimizer for action chunks',
    author='Suhan Park',
    author_email='park94@kw.ac.kr',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.18.0',
        'cvxpy>=1.1.0',
    ],
    python_requires='>=3.7',
    classifiers=[
    ],
)