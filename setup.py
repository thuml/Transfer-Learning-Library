from setuptools import setup, find_packages
import re
from os import path

here = path.abspath(path.dirname(__file__))

# Get the version string
with open(path.join(here, 'dalib', '__init__.py')) as f:
    version = re.search(r'__version__ = \'(.*?)\'', f.read()).group(1)

if __name__ == '__main__':
    setup(
        name="dalib", # Replace with your own username
        version=version,
        author="THUML",
        author_email="JiangJunguang1123@outlook.com",
        keywords="domain adaptation, transfer learning, deep learning, pytorch",
        description="A Library for Deep Domain Adaptation",
        long_description=open('README.md', encoding='utf8').read(),
        long_description_content_type="text/markdown",
        url="https://github.com/thuml/Domain-Adaptation-Lib",
        packages=find_packages(exclude=['docs', 'examples', 'tools']),
        classifiers=[
            # How mature is this project? Common values are
            #   3 - Alpha
            #   4 - Beta
            #   5 - Production/Stable
            'Development Status :: 3 - Alpha',
            # Indicate who your project is intended for
            'Intended Audience :: Science/Research',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
            'Topic :: Software Development :: Libraries :: Python Modules',
            # Pick your license as you wish (should match "license" above)
            'License :: OSI Approved :: MIT License',
            # Specify the Python versions you support here. In particular, ensure
            # that you indicate whether you support Python 2, Python 3 or both.
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
        ],
        python_requires='>=3.6',
        install_requires=[
            'torch>=1.4.0',
            'torchvision>=0.5.0',
            'numpy',
            'qpsolvers>=1.4.0'
        ],
        extras_require={
            'dev': [
                'Sphinx',
                'sphinx_rtd_theme',
            ]
        },
    )
