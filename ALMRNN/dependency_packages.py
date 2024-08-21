# mypackage/install packages what we need
import subprocess

package_dict = {
    'numpy': 'np',
    'pandas': 'pd',
    'scipy': 'scipy',
    'copy': 'copy',
    'time': 'time',
    'random': 'random',
    'math': 'math',
    'gc': 'gc'
}


def check_and_install_packages(package_dict):
    """
    Check if the given packages are installed in the current environment.
    If a package is not installed, it will be installed automatically.
    
    Args:
        packages (list): List of package names.
    """
    for package in package_dict:
        try:
            __import__(package)
            print(f"{package} is already installed.")
        except ImportError:
            print(f"{package} is not installed. Installing...")
            subprocess.call(['pip', 'install', package, '-y'])
            print(f"{package} has been installed.")


# Ensure all required packages are installed
check_and_install_packages(package_dict)

# Directly import the packages with their aliases
import numpy as np
import pandas as pd
import scipy as scipy
import copy as copy
import time as time
import random as random
import math as math
import gc as gc

# Make these imports available to other modules
__all__ = ['np', 'pd', 'scipy', 'copy', 'time', 'random', 'math', 'gc']
