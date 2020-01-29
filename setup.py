from setuptools import setup, find_packages

setup(
    name='ladim_ibm',
    version='0.1',
    packages=find_packages(),
    url='https://git.imr.no/a5606/ladim_ibm',
    license='MIT Licence',
    author='Pål Næverlid Sævik',
    author_email='a5606@hi.no',
    description='IBMs for LADiM', install_requires=['numpy', 'pytest', 'xarray',
                                                    'PyYAML'],
)
