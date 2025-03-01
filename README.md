<p align="center">
    <img src="./media/github_image.png" width="600" />
</p>

## Overview
This script performs an aerodynamic optimization process for airfoils and evaluates their aerodynamic efficiency. It generates airfoil geometries from optimization results, computes aerodynamic performance metrics, and visualizes the aerodynamic efficiency (CL/CD) against the angle of attack.

## Features
- Computes airfoil efficiency using `airfoil_efficiency` function.
- Generates airfoil coordinates using `generate_airfoil_dat_from_points` with optional visualization and smoothing.
- Performs aerodynamic analysis on the generated airfoils using `nf.get_aero_from_coordinates`.
- Plots the aerodynamic efficiency (CL/CD) over a range of angles of attack.
- Visualize the results

## Dependencies
Ensure the following Python libraries are installed:
```bash
pip install numpy matplotlib neuralfoil
```
More information about NeuralFoil can be found at the official repository: [NeuralFoil GitHub](https://github.com/peterdsharpe/NeuralFoil).

## Usage
1. Run the script to:
   - Generate airfoil geometries from optimization results (`result_1.x`, `result_2.x`, `result_3.x`).
   - Compute and plot aerodynamic efficiency over a range of angles of attack.

2. Modify parameters such as `trailing_edge_smooth_factor`, initial conditions (`x0`), Reynolds number (`Re`), and `alpha` range to explore different conditions.

3. New initializations (`x0`) can be generated using the [Onshape File](https://cad.onshape.com/documents/719e17d0922a535027d8e256/w/5e29fd0e1497d5df12605f99/e/da4c96dbb3737816e4659c28?renderMode=0&uiState=67c32f97fba0b20ad7ec9be6) by changing the link lengths under the Variable Table. Note: A copy of the document must be made to edit.   

## Output
- **Airfoil Data**: Airfoil geometries stored in variables `coords_1`, `coords_2`, `coords_3`.
- **Graph**: `initialization_graph.svg` showing the aerodynamic efficiency (CL/CD) vs. angle of attack.