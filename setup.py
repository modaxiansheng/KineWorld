import os
from setuptools import setup, find_packages

requirements_path = os.path.join(os.path.dirname(__file__), "requirements.txt")

if os.path.exists(requirements_path):
    with open(requirements_path, 'r') as f:
        install_requires = [line.strip() for line in f if line.strip() and not line.startswith('#')]
else:
    install_requires = []

setup(
    name="kineworld",
    version="1.0.0",
    description="KineWorld: action-conditioned world model",
    packages=find_packages(),
    install_requires=install_requires,
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    package_data={"diffsynth": ["tokenizer_configs/**/**/*.*"]},
    python_requires='>=3.10',
)
