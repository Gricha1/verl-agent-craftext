from setuptools import setup, find_packages

from pathlib import Path
this_directory = Path(__file__).parent
long_description_content = (this_directory / "README.md").read_text()

setup(
    name='CrafText',
    version='0.1.3',
    description='A text processing package with various scenarios and checkers.',
    author='ZoyaV',
    url='https://github.com/ZoyaV/CrafText',
    packages=find_packages(),
    long_description=long_description_content,
    long_description_content_type='text/markdown',
    install_requires=[
        "absl-py>=1.4.0",
        "anyio>=4.5.0",
        "argon2-cffi>=23.1.0",
        "black>=24.8.0",
        "chex>=0.1.86",
        "craftax>=1.4.3",
        "distrax>=0.1.5",
        "dm-tree>=0.1.8",
        "etils>=0.9.0",
        "flax>=0.8.5",
        "gym>=0.26.2",
        "gymnax>=0.0.6",
        "jax[cuda12]>=0.4.30",
        "jaxlib>=0.4.30",
        "numpy>=1.24.4",
        "optax>=0.2.1",
        "pandas>=1.3.5",
        "scikit-learn>=1.5.2",
        "scipy>=1.10.1",
        "sentence-transformers>=3.1.1",
        "stable-baselines3>=2.0.0",
        "tensorboard>=2.11.2",
        "transformers>=4.44.2",
        "wandb>=0.13.11"
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
    package_data={
        'craftext': ['craftext/dataset/scenarious/*'],  # Adjust this path to where your scenario files are located
    },
)