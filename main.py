import pprint
import numpy as np

from helpers import (
    create_directory_if_not_exists,
    create_mps,
    create_problems_files,
    get_problems,
    solve_all_problems,
)
from run_experiments import run_multipivot_simplex, save_results


def create_mps_files_and_save():
    folder_name = "./multisimplex/Examples/MSPFILES"
    create_problems_files(folder_name)
    solve_all_problems(folder_name)
    print(
        f"Problems Saved in MPS format and Solved and saved in {folder_name}/exact_solutions.xlsx"
    )


def run_indiviual_problem(problem, kwargs):
    Z, feasible, max_RC, stats = run_multipivot_simplex(problem, kwargs)
    pprint.pprint({**stats, "z": Z, "IsFeasible": feasible, "max_RC": max_RC})


def run_save_all(problems, maxiters=1000):
    create_directory_if_not_exists("./results")
    count_solved, total_problem, excel_file_saved = save_results(
        problems, maxiters=maxiters
    )
    print(
        f"{count_solved} problem solved out of ({total_problem}) and results saved in {excel_file_saved}"
    )


if __name__ == "__main__":
    print("welcome to main")
    ## (1) Create MPS files for the problems and solve them using Primal Simplex
    create_mps_files_and_save()

    ## (2) you can run individual experiments as
    # run_indiviual_problem(
    #     19, {"Cut": True, "Order": False, "Pivots": 2, "maxiters": 1000}
    # )

    ## (3) Run and Save to file all experiments
    # run_save_all(25, maxiters=2000)
