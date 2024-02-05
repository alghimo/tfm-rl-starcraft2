from setuptools import setup, find_packages

setup(
    name="tfm_sc2",
    python_requires=">=3.8,<3.9",
    version="0.1.0",
    packages=find_packages(include=["tfm_sc2"]),
    install_requires=["pysc2==4.0.0", "protobuf==3.19.6", "pygame==1.9.6"],
)
