# Vendored: Python implementation of SEISMIC_CPML (Ben Moseley)

The 2D acoustic finite-difference time-domain (FDTD) solver with convolutional
perfectly matched layers (C-PML) used for the FDTD baseline in this project.

| | |
|---|---|
| Source | [benmoseley/FBPINNs](https://github.com/benmoseley/FBPINNs), `fbpinns/traditional_solutions/seismic_cpml/` at commit `fdef915` |
| Author | Ben Moseley |
| License | MIT (see [`LICENSE`](LICENSE), copied unchanged) |
| Algorithm | Python port of `seismic_CPML_2D_pressure_second_order.f90` from [SEISMIC_CPML](https://github.com/geodynamics/seismic_cpml) by Dimitri Komatitsch and Roland Martin (CNRS); that Fortran library is distributed under GPL-3.0 |

## Changes

Both files are copied verbatim, except for one line in
`seismic_CPML_2D_pressure_second_order.py`: the helper is imported as a local
module (`from seismic_CPML_helper import ...`) instead of through the
`fbpinns` package path.

## References

- R. Martin and D. Komatitsch, "An unsplit convolutional perfectly matched layer
  improved at grazing incidence for the seismic wave equation", *Geophysics* 74(5), 2009.
- B. Moseley, A. Markham and T. Nissen-Meyer, "Finite basis physics-informed
  neural networks (FBPINNs)", *Advances in Computational Mathematics*, 2023.
