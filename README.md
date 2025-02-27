# RL project on PPO 

***Gleb Goncharov & Marceau Leclerc - M2DS***

This repository is part of the final project for the 2024 Reinforcement Learning Course. We aim to summarize the [PPO article](https://arxiv.org/abs/1707.06347) and to reproduce some of its experiments.

---

## Environment

To run our scripts and notebooks, we **strongly suggest** using a `python==3.9` environment.  All dependencies can be installed as so :

```sh
pip install -r requirements.txt
```

If using a custom environment, please ensure `mujoco==2.3.7`.

---

## Organization

- `scripts/` hold the python files that were used to train various methods for 1M timesteps. They can all be run using:

  ```sh
  python3 method_torchrl.py
  ```

  These scripts save:

  - Logs (`.csv` files)
  - Models (`.pth` files)
  - Summary plots

  in the `runs/` directory. Experiments are in method-specific subdirectories and all files are named according to the method, name of environment (game) and run number.

- `notebooks/` holds 2 jupyter notebooks that were used to produce some of the figures shown in the report.
