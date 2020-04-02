from setuptools import setup, find_packages

setup(
    name='ladim_plugins',
    version='1.1.0',
    packages=find_packages(),
    package_data={'': ['*.nc', '*.md', '*.rls', '*.yaml', '*.m']},
    url='https://github.com/pnsaevik/ladim_plugins',
    license='MIT Licence',
    author='Pål Næverlid Sævik',
    author_email='a5606@hi.no',
    description='IBMs for LADiM',
    install_requires=[
        'numpy', 'pytest', 'xarray', 'PyYAML', 'netCDF4',
        'triangle', 'scipy', 'cftime<1.1',
    ],
)
