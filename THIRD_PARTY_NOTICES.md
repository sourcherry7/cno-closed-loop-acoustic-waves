# Third-party notices

The code in `cno_closed_loop/` and `scripts/` is my own work under the MIT
License (see [`LICENSE`](LICENSE)). Files adapted from third-party code say so
in their header. This repository also uses or contains the following work.

## Convolutional Neural Operators (CAMLab, ETH Zurich)

- **Used as** the git submodule [`external/CNO`](external/CNO): my fork of
  [camlab-ethz/ConvolutionalNeuralOperator](https://github.com/camlab-ethz/ConvolutionalNeuralOperator),
  branch `closed-loop-waves`, based on upstream commit `75a3b9b`. On top of
  upstream, the branch only adds support for the HDF5 Wave-Gauss dataset.
- **Adapted in** `cno_closed_loop/forward/train.py` and `cno_closed_loop/inverse/`
  (`train.py`, `config.py`, `dataset.py`, `model.py`).
- **Paper:** B. Raonić, R. Molinaro, T. De Ryck, T. Rohner, F. Bartolucci,
  R. Alaifari, S. Mishra, E. de Bézenac, "Convolutional Neural Operators for
  robust and accurate learning of PDEs", NeurIPS 2023.
- **License:** MIT

```
MIT License

Copyright (c) 2024 Computational and Applied Mathematics Laboratory @ ETH Zurich

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## NVIDIA StyleGAN3 code (inside the CNO submodule)

The CNO's alias-free filters and custom CUDA operators (`torch_utils/`,
`training/filtered_networks.py`, and the filter implementation in
`CNO_timeModule_CIN.py`) originate from NVIDIA's StyleGAN3. They are distributed
with the CNO repository under the NVIDIA Source Code License, which limits use to
non-commercial purposes; see
`external/CNO/CNO2d_time_dependent_&_foundation_model/LICENSE_NVIDIA.txt`.
None of this code is copied into this repository's own files.

## FDTD solver: Python implementation of SEISMIC_CPML (Ben Moseley)

- **Vendored in** [`external/seismic_cpml`](external/seismic_cpml), copied from
  [benmoseley/FBPINNs](https://github.com/benmoseley/FBPINNs) at commit `fdef915`
  (verbatim except for one import line).
- **License:** MIT, Copyright (c) 2021 Ben Moseley; full text in
  [`external/seismic_cpml/LICENSE`](external/seismic_cpml/LICENSE).
- **Origin of the algorithm:** [SEISMIC_CPML](https://github.com/geodynamics/seismic_cpml)
  by Dimitri Komatitsch and Roland Martin (GPL-3.0); R. Martin and D. Komatitsch,
  *Geophysics* 74(5), 2009.

## Wave-Gauss dataset generator (Bogdan Raonić)

The training data were generated with an unpublished data-generation script by
Bogdan Raonić, which I adapted and parallelized for this project. The script is
not part of this repository.
