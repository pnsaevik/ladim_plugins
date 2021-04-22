# Utility functions

This module contains a range of utility functions to be used by IBMs.

## Light model

The model is based on [Skartveit, A. & Olseth, J.A. (1988)](http://web.gfi.uib.no/publikasjoner/rmo/RMO-1988-7.pdf).
"Varighetstabeller for timevis belysning mot 5 flater på 16 norske stasjoner". Meteorological
report series 1988-7, University of Bergen.

The model computes solar irradiation as a function of time, longitude, latitude, depth and
light extinction coefficient.

Usage: `EB = light(time, lon, lat, depth, extinction_coefficient)`

Depth is given in m, extinction coefficient in m^-1. Default values are depth = 0 and
extinction_coefficient = 0.2. Return value is given in µmol photons s^-1 m^-2.


## Density model

Computes 1-atm seawater density from temperature and salinity. Reference: Fofonoff, N.P. &
Millard, R.C. (1983). "Algorithms for Computation of Fundamental Properties of Seawater".
Unesco Techical Papers in Marine Science.

Usage: `rho = density(salt, temp)`

Salinity is given in PSU and temperature is given in degrees Celcius. Return value is given
in kg m^-3.
 

## History

Created by Pål Næverlid Sævik (2021)
