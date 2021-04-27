from setuptools import setup, find_packages

setup(
    name='ladim_plugins',
    version='1.3.5',
    packages=find_packages(),
    package_data={'': [
        '*.nc', '*.md', '*.rls', '*.yaml', '*.m', '*.bsp', '*.geojson',
    ]},
    entry_points={
        'console_scripts': ['makrel=ladim_plugins.release.makrel:main'],
    },
    url='https://github.com/pnsaevik/ladim_plugins',
    license='MIT',
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 5 - Production/Stable',

        # Indicate who your project is intended for
        'Intended Audience :: Science/Research',
        'Topic :: Software Development :: Build Tools',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',

        "Operating System :: OS Independent",
    ],
    author='Pål Næverlid Sævik',
    author_email='a5606@hi.no',
    description='Plugins for LADiM',
    install_requires=[
        'numpy', 'pytest', 'xarray', 'PyYAML', 'netCDF4', 'pyproj',
        'triangle', 'scipy', 'cftime', 'ladim', 'skyfield', 'pandas', 'requests'
    ],
    python_requires='>=3.6',
)
