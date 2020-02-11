from setuptools import setup, find_packages

setup(
    name='ladim_plugins',
    version='0.1.0',
    packages=find_packages(),
    package_data={'': ['*.nc', '*.md', '*.rls', '*.yaml', '*.m']},
    url='https://git.imr.no/a5606/ladim_ibm',
    license='MIT Licence',
    author='Pål Næverlid Sævik',
    author_email='a5606@hi.no',
    description='IBMs for LADiM', install_requires=['numpy', 'pytest', 'xarray',
                                                    'PyYAML', 'netCDF4'],
)
