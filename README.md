# Project 5.3: Minimization of Weakly Convex Functions

Task: Optimize weakly convex functions, implementing stochastic proximal-point-like steps and theoretically tracking the convergence of the gradient of the Moreau envelope.

## Literature

Davis, D., & Drusvyatskiy, D. (2019). Stochastic model-based minimization of weakly convex functions. SIAM Journal on Optimization, 29(1), 207-239.

## Environment setup

1. Make sure you have conda cli installed on your machine
2. Create conda environment:

    ```bash
    conda create -n minimization-of-weakly-convex-functions python=3.13 poetry=2.4.1
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

## Running experiments

All experiment scripts are in `experiments/`. Activate the conda environment first:

```bash
conda activate minimization-of-weakly-convex-functions
```

On macOS with Apple Silicon you may need to set `KMP_DUPLICATE_LIB_OK=TRUE` to avoid an OpenMP conflict.

### Experiment 1 – Phase Retrieval (Figure 3)

Compares three stochastic model-based methods (subgradient, prox-linear, proximal point)
across 100 step-sizes on the phase retrieval problem. Configs: (d,m) = (10,30), (50,150), (100,300).

```bash
KMP_DUPLICATE_LIB_OK=TRUE python experiments/phase_retrieval.py
```

### Experiment 2 – Blind Deconvolution (Figure 4)

Same comparison on the blind deconvolution problem.
Configs: (d₁,d₂,m) = (10,10,30), (50,50,200), (100,100,400).

```bash
KMP_DUPLICATE_LIB_OK=TRUE python experiments/blind_deconvolution.py
```

### Experiment 3 – Optimizer Comparison

Compares the three model-based methods against SGD, Adam, and AdaGrad on phase retrieval,
showing step-size sensitivity.

```bash
KMP_DUPLICATE_LIB_OK=TRUE python experiments/optimizer_comparison.py
```

### Experiment 4 – Moreau Envelope Gradient Convergence

Tracks ‖∇φ_λ(xₜ)‖ (Moreau envelope gradient norm) over epochs for each method,
demonstrating the paper's main convergence guarantee.

```bash
KMP_DUPLICATE_LIB_OK=TRUE python experiments/moreau_convergence.py
```

All plots are saved to `deliverables/figures/`.
