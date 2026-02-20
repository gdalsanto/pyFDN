# Examples

Notebooks and scripts demonstrating pyFDN features.

## SDN (Scattering Delay Network)

- **`example_sdn_coefficients.ipynb`** — Full walkthrough: room geometry and wall filters → SDN `compute()` → `sdn.sdn_to_flamo()` → FLAMO impulse response and 3D visualization.  
  The script `example_sdn_coefficients.py` runs the same pipeline from the command line.

## Other examples

- **Vanilla FDN** — `example_vanilla_FDN.ipynb`: build and alter a vanilla FDN, plot IRs.
- **Absorption** — `example_absorption_filters.ipynb`, `example_one_pole_absorption.ipynb`: absorption filters and RT.
- **Coupled rooms** — `example_coupled_rooms.ipynb`: coupled FDN rooms.
- **Matrix / delays** — `example_interpolate_matrix.ipynb`, `example_delay_matrix_density.ipynb`, `example_colorless_FDN.ipynb`.
- **Processing** — `example_process_fdn.ipynb`, `example_zFilter.ipynb`, `example_dss2ss.ipynb`.
