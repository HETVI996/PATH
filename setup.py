from setuptools import setup, find_packages
from typing import List

HYPHEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    '''
    Reads the requirements from the given file and returns them as a list.
    
    '''
    
    requirements = []
    with open(file_path, 'r') as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.strip() for req in requirements]
        
        # Remove -e . if present
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
            
    return requirements

setup(
    name = "Path",
    version = "0.0.1",
    author = "Hetvi Kakkad",
    author_email="kakkadhetvi87@gmail.com",
    packages = find_packages(),
    install_requires = get_requirements('requirements.txt')
)