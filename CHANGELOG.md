# Changelog

All notable changes to the project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.7.0] - 2025-08-13
### Added
- New jitter option in sedimentation module

## [2.6.1] - 2025-08-13
### Fixed
- Thredds module download cache invalidation now works

## [2.6.0] - 2025-08-13
### Added
- Thredds module, which allows the use of a thredds server to provide
  s-coordinate netCDF files as forcing.

## [2.5.2] - 2025-05-20
### Fixed
- Gridforce module in chemicals plugin now inherits from main ladim gridforce.
  This allows particles near the boundary of the model area.

## [2.5.1] - 2025-03-28
### New
- Added stub VPS module
### Fixed
- Rename multiplicity parameter in salmon lice module so that it is consistent

## [2.4.1] - 2025-03-03
### Changed
- Salmon lice particles now dies at 170 degree-days instead of 200

## [2.4.0] - 2025-01-29
### Changed
- Updated gridforce modules of "chemicals" and "nk800met" to facilitate
  future Ladim improvements

## [2.3.2] - 2024-12-10
### Added
- Saithe module with individual directional behaviour

## [2.2.0] - 2024-09-25
### Added
- Lifespan option to chemicals module 

## [2.1.0] - 2024-09-13
### Changed
- Salmon lice model now differentiates between different planktonic life stages. Salinity preference settings are tuned
  to match observed vertical distribution of lice when NorKyst v3 is used as forcing. 

## [2.0.4] - 2024-04-25
### Fixed
- Minor error in sedimentation and mine yaml files

## [2.0.3] - 2024-04-25
### Added
- Sedimentation module: Allow skipping vertical diffusion

## [2.0.2] - 2024-04-25
### Added
- Sedimentation module: Allow skipping resuspension check

## [2.0.1] - 2024-04-24
### Added
- Sedimentation module: Auto sinking velocity if unspecified at release

## [2.0.0] - 2024-03-20
### Changed
- Updated examples so that they work with Ladim v2

## [1.9.5] - 2024-03-20
### Changed
- Modifications to IBM's to comply with new Ladim API format

## [1.9.4] - 2024-01-11
### Fixed
- Updated various deprecated code in modules and test framework
- Package now works for python 3.12

## [1.9.3] - 2023-10-24
### Added
- Auto upload to PyPI  

## [1.9.2] - 2023-08-28
### Changed
- Migrate from CircleCI to GitHub Actions  

## [1.9.1] - 2023-04-19
### Added
- Utility function for converting output to sqlite3 format 

## [1.9.0] - 2023-01-24
### Added
- Changelog document
### Fixed
- In module `shrimp`: Bug where forcing variables were not updated 

## [1.8.0] - 2022-11
### Added
- Shrimp module

## [1.7.1] - 2022-11
### Added
- In module `salmon_lice`: Function to compute salmon lice infectivity 

## [1.6.5] - 2022-09
### Added
- In module `sandeel`: Add hatching and metamorphose behaviour

## [1.6.4] - 2022-06
### Fixed
- Bug in `nk800met` module

## [1.6.3] - 2022-06
### Fixed
- Bug in `mine` module

## [1.6.2] - 2022-05
### Added
- In module `release`: Add attributes from geojson files


## [1.6.0] - 2022-04
### Added
- Module `sandeel`

## [1.5.10] - 2021-12
### Added
- In module `chemicals`: Added coastal hyperdiffusion as a land collision avoidance method

## [1.5.9] - 2021-11
### Fixed
- In module `release`: Let output be sorted by release date, as ladim requires
  date sorting to work properly

## [1.5.8] - 2021-09
### Added
- In module `chemicals`: Added heterogeneous diffusion

## [1.4.0] - 2021-05
### Added
- Module `salmon_lice` 
- Module `utils`
- In module `release`: Allow attributes generated from statistical distributions
- In module `sedimentation`: Linear interpolation of bathymetry

## [1.3.0] - 2020-10
### Added
- Module `release`
- Module `nk800met`

## [1.2.2] - 2020-10
### Changed
- In module `chemicals`: Memory optimizations
### Fixed
- In module `chemicals`: Corrections to vertical velocity computation

## [1.2.1] - 2020-09
### Fixed
- Corrections to module `sedimentation`

## [1.2] 2020-04
### Added
- Module `lunar_eel`

## [1.1] - 2020-04
### Added
- In module `sedimentation`: Resuspension mechanics

## [1.0] - 2019-12
### Added
- Module `chemicals`
- Module `egg`
- Module `salmon_lice`
- Module `sedimentation`
