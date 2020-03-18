from setuptools import find_packages, setup


with open("README.md", "r") as fh:
    long_description = fh.read()


if __name__ == '__main__':
    setup(
        name="dalib", # Replace with your own username
        version="0.0.1",
        author="THUML",
        author_email="13126830206@163.com",
        keywords="domain adaptation, transfer learning, deep learning",
        description="Deep Domain Adaptation Library",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/thuml/Domain-Adaptation-Lib",
        packages=find_packages(),
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
        ],
        python_requires='>=3.6',
        install_requires=[  # 添加了依赖的 package
            'pytorch>=1.4',
            'torchvision',
            'numpy'
        ]
    )
