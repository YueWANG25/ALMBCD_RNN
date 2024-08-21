from setuptools import setup, find_packages

setup(
    name='ALMRNN',
    version='0.1.0',
    description='A Python package for ALM_BCD with ReLU and ELU for training RNNs',
    author='Yue Wang, Chao Zhang, Xiaojun Chen',
    author_email='yueyue.wang@connect.polyu.hk',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.26.4',
        'pandas>=2.2.2',
        'scipy>=1.13.1'
        # Add other dependencies here
    ],
    extras_require={
        # 'dev': [
        #     'pytest>=6.2.4',
        #     'black>=21.7b0',
        #     # Add other development dependencies here
        # ]
    },
    python_requires='>=3.9.18',
)