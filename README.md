# Catheter Utils

This project contains utils related to our catheter tracking projects.

## `cathcoords`

Contains things for manipulating "cathcoords" files and data.
These are localized catheter coordinates combined with metadata like trigger time and snr.

## `geometry`

Contains things for reasoning about the catheter's geometry.
Things like distances between distal, proximal, and tip locations, as well as helpers for extrapolating the tip.

## `localization`

Contains things for producing estimates of a coil location from projection data.
This includes single projection localization all the way up to full catheter location fit using constrains
and snr-like error weighting.

## `projections`

Contains code for reading raw projection data, as well as basic calculations like snr.

## Tests

Can run with the command `python -m unittest discover tests`.
