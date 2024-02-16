from setuptools import setup, find_packages

setup(
    name="tfm_sc2",
    python_requires=">=3.8",
    version="0.1.0",
    packages=find_packages(include=["tfm_sc2"]),
    install_requires=["pysc2==4.0.0", "protobuf==3.19.6"],
)
