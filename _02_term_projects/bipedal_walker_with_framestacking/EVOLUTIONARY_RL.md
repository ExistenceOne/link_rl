# Evolutionary RL on BipedalWalkerHardcore-v3 (ERL & CEM-RL)

This directory adds two families of **evolution + reinforcement-learning hybrids** for the three
off-policy continuous-control algorithms used in the course (SAC, DDPG, TD3), all running on
`BipedalWalkerHardcore-v3` with an optional frame-stacking wrapper:

| Folder          | Method  | Base RL | Actor type              |
|-----------------|---------|---------|-------------------------|
| `erl-sac/`      | ERL     | SAC     | stochastic `GaussianPolicy` |
| `erl-ddpg/`     | ERL     | DDPG    | deterministic `Actor`   |
| `erl-td3/`      | ERL     | TD3     | deterministic `Actor`   |
| `cem-rl-sac/`   | CEM-RL  | SAC     | stochastic `GaussianPolicy` |
| `cem-rl-ddpg/`  | CEM-RL  | DDPG    | deterministic `Actor`   |
| `cem-rl-td3/`   | CEM-RL  | TD3     | deterministic `Actor`   |

Both methods maintain a **real population of additional actor networks** whose fitness comes from
**actual environment rollouts** (episodic return). This is a deliberate departure from the earlier
critic-surrogate ES in `_02_term_projects/bipedal_walker/sac/b_sac_train.py` (`_es_update`), which
only perturbed a single policy in place and used `min(Q1,Q2)` as a surrogate fitness — it never
instantiated a separate policy network. That pattern was **intentionally discarded** here.

Each folder is self-contained and follows the repo convention: an `a_*_models.py` (networks +
replay buffer) and a `b_*_train.py` (agent + training loop). Run a trainer from inside its folder.

---

## Shared design (all six folders)

- **Environment**: `BipedalWalkerHardcore-v3` (obs `(24,)`, 4 continuous actions in `[-1, 1]`).
- **Frame-stacking toggle**: config key `stack_size`.
  - `stack_size > 1` wraps the env with `gymnasium.wrappers.FrameStackObservation` → obs `(stack, 24)`.
  - `stack_size == 1` disables stacking → obs `(24,)`.
  - Networks size to `n_features = prod(obs_shape)` and flatten with `flatten(start_dim=-obs_ndim)`,
    so the same code handles both shapes (`obs_ndim = len(obs_shape)`).
- **Replay buffer**: a single GPU-resident tensor `ReplayBuffer` (pre-allocated; arbitrary
  `observation_shape`) shared by the whole population and the RL learner.
- **Actor networks**: 256-hidden MLPs; deterministic `Actor` outputs `tanh ∈ [-1,1]` matching the
  action space directly (no `*2` scaling). DDPG/TD3 exploration is Gaussian (`exploration_noise`, default 0.1).
- **Fitness**: episodic return averaged over `eval_episodes` real rollouts.

---

## Method 1 — ERL (Evolution-Guided Policy Gradient, Khadka & Tumer 2018)

A GA-evolved population of actors **and** one gradient-based RL learner share a replay buffer.

Per generation:
1. **Population rollouts** — each of `pop_size` actors is evaluated deterministically in the env;
   fitness recorded; all transitions pushed to the shared buffer.
2. **RL actor rollout** — the learner's actor runs with exploration (SAC: stochastic sample;
   DDPG/TD3: Gaussian noise); transitions pushed to the buffer.
3. **RL gradient updates** — `int(steps_collected * grad_steps_ratio)` standard updates of the
   underlying algorithm from the buffer.
4. **Evolution** — rank by fitness; keep the top `n_elite` (elitism); fill the rest by
   tournament selection (`tournament_k`) → uniform per-weight crossover → Gaussian mutation
   (`mut_prob`, `mut_strength`).
5. **RL → population injection** — every `sync_period` generations the RL actor's weights overwrite
   a non-elite population slot, re-injecting gradient-learned behaviour into the gene pool.

Per-algorithm notes:
- **erl-sac**: population members are `GaussianPolicy` evaluated at the mean action; RL part is full
  SAC (twin soft-Q, automatic entropy α). The GA perturbs all params including the mean/log_std heads.
- **erl-ddpg**: deterministic actors; RL part is DDPG (single critic, deterministic policy gradient).
- **erl-td3**: deterministic actors; RL part is TD3 (twin critic, target-policy smoothing, delayed
  actor updates).

---

## Method 2 — CEM-RL (Pourchot & Sigaud 2019)

A diagonal Cross-Entropy-Method distribution `N(mean, diag(var))` over the **flattened actor
parameters** drives the population; a shared critic injects RL gradients into half of it.

Per generation:
1. **Sample** `pop_size` actors: `θ_i = mean + sqrt(var + cem_noise) ⊙ N(0, I)`.
2. **Gradient half** — for the first `pop_size//2` actors, load `θ_i` into the gradient container,
   apply `n_grad_steps` updates using the shared critic (critic is trained from the buffer over those
   steps), and write the improved params back. The other half are left as sampled.
3. **Evaluate all** members in the env (deterministic) → fitness; transitions pushed to the buffer.
4. **CEM refit** — take the top `n_elite` (top half) and update `mean`/`var` with CMA-ES-style rank
   weights (variance computed against the *old* mean).
5. **Noise decay** — the exploration-noise floor `cem_noise` decays each generation
   (`noise_init → noise_end` at rate `noise_decay`).

Validation and checkpoints use the current CEM **mean** actor (deterministic).

Per-algorithm notes:
- **cem-rl-ddpg / cem-rl-td3**: CEM over deterministic `Actor` params; the gradient half uses the
  DPG actor loss. TD3 adds twin-min targets, target-policy smoothing, and delayed policy updates.
- **cem-rl-sac**: CEM over `GaussianPolicy` params (mean & log_std heads); the gradient half uses the
  SAC actor objective with the entropy term, and the automatic temperature `alpha` is shared globally.
  SAC has no target actor — TD targets use the current actor.

---

## ERL vs CEM-RL at a glance

| Aspect              | ERL                                   | CEM-RL                                        |
|---------------------|---------------------------------------|-----------------------------------------------|
| Population source   | explicit set of actor networks        | samples of a Gaussian over flat params        |
| Variation operator  | tournament + crossover + mutation     | CEM mean/variance refit on elites             |
| RL → population path | periodic weight injection             | gradient steps on half the sampled population |
| RL learner          | one persistent actor + critic         | one gradient container + persistent critic    |
| Fitness             | real env rollouts                     | real env rollouts                             |

---

## Hyperparameters (defaults in each `b_*_train.py` `main()`)

Common: `stack_size=4`, `gamma=0.99`, `batch_size=256`, `replay_buffer_size=1_000_000`,
`soft_update_tau=0.995`, learning rate(s) `3e-4`, `learning_starts=10_000`,
`validation_num_episodes=3`, `eval_episodes=1`, `pop_size=10`.
- ERL: `n_elite=2`, `tournament_k=3`, `mut_prob=0.1`, `mut_strength=0.1`, `sync_period=1`,
  `grad_steps_ratio=1.0`; DDPG/TD3 add `exploration_noise=0.1`.
- CEM-RL: `n_elite=5` (top half), `n_grad_steps=200`, `sigma_init=1e-3`,
  `noise_init=1e-3`, `noise_end=1e-5`, `noise_decay=0.999`.
- TD3 (both methods): `policy_update_delay=2`, `target_policy_noise=0.2`, `target_policy_noise_clip=0.5`.
- SAC (both methods): `automatic_entropy_tuning=True`.

Population evaluation is single-process and is the main wall-clock cost. On the RTX 3060 / H200 boxes
the tensor replay buffer stays on the GPU (≈0.8 GB for 1e6 stacked transitions). To go lighter, reduce
`pop_size`, `replay_buffer_size`, or (for CEM-RL) `n_grad_steps`.

---

## How to run

```bash
cd erl-sac        # or any of the six folders
python b_erl_sac_train.py
```

Each trainer logs to Weights & Biases by default (project e.g. `erl_sac_BipedalWalkerHardcore-v3`,
`cem_rl_td3_BipedalWalkerHardcore-v3`). Set `use_wandb = False` in `main()` to disable. Best models
are written to that folder's `models/` as `<method>_<algo>_<env>_<reward>_<time>.pth` plus a
`<method>_<algo>_<env>_latest.pth` copy. To disable frame stacking, set `"stack_size": 1` in the config.
```
