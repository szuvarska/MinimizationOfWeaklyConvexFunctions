# Project 5.3: Minimization of Weakly Convex Functions

Task: Optimize weakly convex functions, implementing stochastic proximal-point-like steps and theoretically tracking the convergence of the gradient of the Moreau envelope.

## Environment setup

1. Make sure you have conda cli installed on your machine
2. Create conda environment:

    ```bash
    conda create -n minimization-of-weakly-convex-functions python=3.13 poetry
    ```

3. Setup poetry:

    ```bash
    poetry install
    ```

4. Install pre-commit hooks

    ```bash
    poetry run pre-commit install --hook-type commit-msg
    ```

5. Add packages to poetry using:

    ```bash
    poetry add package-name
    ```

## Project structure

- `src/` – core algorithms
- `experiments/` – scripts for running experiments
- `notebooks/` – exploratory analysis
- `deliverables/` - presentations reports etc.
