from setuptools import setup, find_packages


if __name__ == '__main__':
    setup(
        name="dalib", # Replace with your own username
        version="0.0.1",
        author="THUML",
        author_email="JiangJunguang1123@outlook.com",
        keywords="domain adaptation, transfer learning, deep learning, pytorch",
        description="A Library for Deep Domain Adaptation",
        long_description=open('README.md', encoding='utf8').read(),
        long_description_content_type="text/markdown",
        url="https://github.com/thuml/Domain-Adaptation-Lib",
        packages=find_packages(include=['dalib']),
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
        install_requires=[  # 添加了依赖的 package
            'torch>=1.4.0',
            'torchvision',
            'numpy'
        ],
        extras_require={
            'dev': [
                'Sphinx',
                'sphinx_rtd_theme',
            ]
        },
    )
