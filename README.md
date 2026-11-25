# RL Robotic Arm — Two-Timescale & Normalization PPO Ablations on `Pusher-v5`

**A from-scratch PyTorch implementation and empirical ablation suite evaluating two-timescale stochastic approximation, observation normalization, and advantage normalization for Proximal Policy Optimization (PPO) on the 7-DoF MuJoCo `Pusher-v5` robotic manipulation task.**

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Gymnasium-MuJoCo-0081A5" alt="Gymnasium MuJoCo">
  <img src="https://img.shields.io/badge/Algorithm-Two--Timescale_PPO-8A2BE2" alt="Two-Timescale PPO">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="MIT License">
</p>

## Table of Contents

- [Overview](#overview)
- [Environment & Dynamics](#environment--dynamics)
- [Ablation Study Design](#ablation-study-design)
- [Mathematical Formulation & Implementation](#mathematical-formulation--implementation)
- [Hyperparameters](#hyperparameters)
- [Results](#results)
- [Installation & Usage](#installation--usage)
- [Project Structure](#project-structure)
- [Created by](#created-by)

## Overview

In continuous-control robotic manipulation, policy gradient stability depends heavily on how quickly the critic tracks shifting state-value targets and how heterogeneous kinematic features are conditioned before entering neural function approximators. Rather than treating PPO as a black box, this repository isolates and benchmarks four architectural and optimization configurations from scratch on MuJoCo's `Pusher-v5`:

1. **`baseline_ppo`**: Standard single-timescale Adam updates with Welford running observation normalization and batch advantage standardization.
2. **`two_timescale_ppo`**: Decoupled polynomial Robbins–Monro step-size schedules forcing the critic to converge on a faster timescale than the actor.
3. **`no_obs_norm`**: Disables running observation mean/variance standardization to measure sensitivity to unscaled joint angles, velocities, and Cartesian coordinates.
4. **`no_adv_norm`**: Passes raw Generalized Advantage Estimates (GAE-$\lambda$) directly into the clipped surrogate objective to isolate gradient-scale sensitivity.

## Environment & Dynamics

The experiments run on Gymnasium's [`Pusher-v5`](https://gymnasium.farama.org/environments/mujoco/pusher/) (MuJoCo physics engine), where a 7-DoF robotic arm must reach a cylindrical puck on a tabletop and push it to a target goal position within a fixed horizon of $T = 100$ steps.

| Component | Dimension / Range | Description |
|-----------|-------------------|-------------|
| **Observation Space** $\mathcal{S}$ | $\mathbb{R}^{23}$ | 7 joint angles, 7 joint angular velocities, 3D fingertip position, 3D cylinder object position, and 3D goal position |
| **Action Space** $\mathcal{A}$ | $[-2.0, 2.0]^7$ | Continuous joint torques applied across shoulder, upper-arm, forearm, and wrist actuators |
| **Reward Function** $R_t$ | $(-\infty, 0]$ | Dense composite reward penalizing fingertip-to-object distance, object-to-goal distance, and squared control effort ($\Vert{}a_t\Vert{}_2^2$) |
| **Termination / Truncation** | $T = 100$ | No early failure termination; episodes truncate strictly at 100 steps, requiring proper value bootstrapping at $s_T$ |

## Ablation Study Design

| Variant | Actor LR ($\alpha_k$) | Critic LR ($\beta_k$) | Obs Norm | Adv Norm | Core Hypothesis Tested |
|---------|-----------------------|-----------------------|----------|----------|------------------------|
| **`baseline_ppo`** | Constant ($3 \times 10^{-4}$) | Constant ($3 \times 10^{-4}$) | Enabled | Enabled | Reference performance with coupled step sizes and full variance-reduction heuristics |
| **`two_timescale_ppo`** | Polynomial decay ($\alpha_k \propto k^{-a}$) | Polynomial decay ($\beta_k \propto k^{-b}$) | Enabled | Enabled | Enforcing $\alpha_k / \beta_k \to 0$ reduces critic tracking lag and prevents policy oscillation during contact-rich pushing |
| **`no_obs_norm`** | Constant ($3 \times 10^{-4}$) | Constant ($3 \times 10^{-4}$) | **Disabled** | Enabled | Unscaled multi-unit state inputs degrade MLP gradient conditioning and slow early reaching |
| **`no_adv_norm`** | Constant ($3 \times 10^{-4}$) | Constant ($3 \times 10^{-4}$) | Enabled | **Disabled** | Raw advantage variance destabilizes trust-region clipping ratios as reward magnitude shifts between phases |

### 1. Two-Timescale Stochastic Approximation (`two_timescale_ppo`)
Standard Actor-Critic implementations update policy and value parameters with identical step sizes. Under stochastic approximation theory, asymptotic convergence requires the critic to operate on a asymptotically faster timescale than the actor so that $V_\psi(s)$ closely approximates the true value function $V^{\pi_\theta}(s)$ of the slowly drifting policy. We instantiate separate PyTorch optimizers and `LambdaLR` schedulers governed by Robbins–Monro conditions:

```math
\sum_{k=0}^{\infty} \alpha_k = \sum_{k=0}^{\infty} \beta_k = \infty, \qquad \sum_{k=0}^{\infty} (\alpha_k^2 + \beta_k^2) < \infty, \qquad \lim_{k \to \infty} \frac{\alpha_k}{\beta_k} = 0
