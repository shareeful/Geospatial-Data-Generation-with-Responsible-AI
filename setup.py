from setuptools import setup, find_packages

setup(
    name="rai_geospatial",
    version="1.0.0",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "torch>=2.0.0",
        "opacus>=1.4.0",
        "rasterio>=1.3.0",
        "geopandas>=0.14.0",
        "pyproj>=3.6.0",
        "scipy>=1.11.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.13.0",
        "pyyaml>=6.0",
        "tqdm>=4.66.0",
        "shapely>=2.0.0",
        "joblib>=1.3.0",
    ],
    entry_points={
        "console_scripts": [
            "rai-geospatial=main:main",
        ],
    },
)
