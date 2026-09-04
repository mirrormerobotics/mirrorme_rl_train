from setuptools import find_packages, setup

setup(
    name="mirrorme-rl-train",
    version="2.1.0",
    description="MIRRORME RL Train Isaac Lab flat-ground locomotion framework for BPX",
    author="MIRRORME Technology",
    license="BSD-3-Clause",
    packages=find_packages(),
    python_requires=">=3.11,<3.12",
    install_requires=["gymnasium", "numpy>=1.26"],
    include_package_data=True,
    zip_safe=False,
)
