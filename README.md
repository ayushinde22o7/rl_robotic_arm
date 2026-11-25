# Proximal Policy Optimization (PPO) Ablation Study on MuJoCo `Pusher-v5`

From-scratch PyTorch implementation and empirical ablation study of Proximal Policy Optimization (PPO) on the continuous-control MuJoCo `Pusher-v5` robotic manipulation environment.

## Overview

This repository investigates the optimization dynamics and stability mechanisms of Actor-Critic policy gradient methods in high-dimensional continuous action spaces. Specifically, the experiments evaluate:
- **Two-Timescale Stochastic Approximation:** Separating actor and critic learning rate schedules under Robbins–Monro decay conditions to stabilize value estimation relative to policy updates.
- **Normalization Ablations:** Isolating the impact of running observation normalization and batch advantage normalization on episodic return and critic value error.

## Repository Structure

```text
rl_robotic_arm/
├── rl_robotic_arm.ipynb               # Complete PPO implementation, ablation runner, and plotting pipeline
├── pusher_v5_reward_ablation.png      # Episodic return comparison across ablation variants
├── pusher_v5_critic_mse_ablation.png  # Critic value-function MSE across training updates
├── videos/                            # Recorded evaluation rollouts for trained policies
└── README.md
