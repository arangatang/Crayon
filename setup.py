from setuptools import setup, find_packages


setup(
    name="crayon",
    packages=find_packages("."),
    include_package_data=True,
    install_requires=[
        "boto3",
        "pandas",
        "scipy",
        "gluonts",
        "sagemaker",
        "seaborn",
        "beautifultable",
        "requests"
    ],
    # entry_points={},
)
