# README.md

## MultiPivotSimplex Solver

This repository provides tools to solve linear programming problems using a Multi-Pivot Simplex method. The solution includes utilities for creating problems in **MPS format**, running individual experiments, and saving results for analysis.

---

## Features
1. **Create MPS Files**: Generate problems in the standardized **MPS format** for use with optimization solvers.
2. **Run Individual Experiments**: Solve a single problem using the Multi-Pivot Simplex method with custom configurations.
3. **Batch Processing**: Run and save results for all problems in a batch to an Excel file for analysis.

---

## Requirements

### Dependencies
This project requires Python 3.7+ and the following libraries:
- `numpy`
- `pprint`
- `helpers` (custom utility module)
- `run_experiments` (custom module)

Install dependencies using:
```bash
pip install numpy
```

---

## Usage

### 1. Generate MPS Files and Solve Problems
This step creates problems in MPS format and solves them using the **Primal Simplex** method. The results are saved in `./multisimplex/Examples/MSPFILES/exact_solutions.xlsx`.

```python
create_mps_files_and_save()
```

### 2. Run Individual Experiments
Run a specific problem with custom configurations, such as enabling/disabling cuts, setting pivot strategies, and defining maximum iterations.

Example:
```python
run_indiviual_problem(
    19, {"Cut": True, "Order": False, "Pivots": 2, "maxiters": 1000}
)
```

### 3. Batch Processing and Saving Results
Run all problems in a batch with a specified number of maximum iterations and save results in an Excel file.

Example:
```python
run_save_all(25, maxiters=2000)
```

This saves results in the `./results` directory.

---

## File Structure

### Main Script
`main.py`:
- The primary entry point for the project.
- Demonstrates how to generate MPS files, run experiments, and save results.

### Helper Functions
`helpers.py`:
- Contains utility functions for:
  - Creating directories.
  - Creating MPS files.
  - Solving problems.
  - Managing problem files.

### Run Experiments
`run_experiments.py`:
- Includes methods to execute Multi-Pivot Simplex experiments and save results.

---

## Example Workflow
1. **Generate and Solve Problems**:
   Run the following command to create MPS files and solve the problems:
   ```bash
   python main.py
   ```

2. **Run Individual Problems**:
   Uncomment and modify the `run_indiviual_problem` function in `main.py` to experiment with a specific problem.

3. **Save Results for All Problems**:
   Uncomment and run the `run_save_all` function in `main.py` to solve all problems and save results.

---

## Output
- **MPS Files**: Saved in `./multisimplex/Examples/MSPFILES/`.
- **Excel Results**: Found in `./results/` (e.g., `exact_solutions.xlsx`).
- **Console Logs**: Provides detailed statistics about the problem-solving process.

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

For any questions or issues, feel free to reach out!