# MCTS Motion Planning with Passively Deformable Obstacles

A proof-of-concept motion planning project developed for my **Georgia Tech Robotics Ph.D. Qualifying Exam (2024)**.

This project explores motion planning in environments containing **passively deformable obstacles**—objects that do not move or deform on their own, but can change state through interaction with the robot. Examples include tubing and wires in clinical environments, clothes in a closet, or soft branches encountered by a field robot.

Unlike conventional obstacle avoidance, contact with such objects is neither always forbidden nor completely unconstrained. Limited interaction may open a feasible route, while excessive deformation may become unsafe.

## Motivation

Most motion planners for dynamic environments focus on obstacles that move or deform independently. Passively deformable obstacles introduce a different problem: a feasible path may not exist until the robot physically interacts with the environment.

The goal here is **not** to manipulate an obstacle into a desired configuration. Instead, the planner seeks a safe robot motion while allowing incidental environment deformation when useful and avoiding hazardous interaction.

The broader qualifying-exam proposal considered combining:

* global motion planning to provide a coarse route;
* local Monte Carlo Tree Search (MCTS) for interaction-aware planning;
* online replanning when deformation limits are reached;
* belief-space reasoning and learned models for uncertain obstacle dynamics.

This repository implements a simplified local-planning prototype of that idea.

## Prototype

The implemented demo considers a **2D non-holonomic point robot** navigating among static, moving, and passively deformable obstacles.

The robot uses a discrete action space controlling forward motion and steering. At every control step, an **MCTS-based local planner** searches over candidate action sequences and predicts how both the robot and nearby deformable obstacles will evolve.

A simple interaction model allows deformable obstacles to be pushed up to a prescribed deformation limit. As a result, the robot may:

* push an obstacle and continue when the interaction opens a feasible route;
* stop interacting when further deformation becomes unsafe;
* retreat from a locally trapped route and replan toward an alternative path.

The prototype uses a local escape heuristic to help the robot leave local minima. The implemented demo does **not** include the global planner proposed in the broader qualifying-exam framework.

## MCTS Planner

The planner follows the standard MCTS structure:

1. **Selection** using an upper-confidence-bound criterion;
2. **Expansion** over the robot's discrete actions;
3. **Predictive environment update** along the candidate robot trajectory;
4. **Heuristic evaluation** of the resulting state instead of a long random rollout;
5. **Backpropagation** of the evaluated reward.

A fixed planning-time budget is used at each control step, making the planner suitable for online replanning in the simulated environment.

The evaluation considers factors including progress toward the goal, obstacle proximity, interaction with deformable obstacles, and local trapping.

## Results

In the experiment used for the qualifying exam:

* **50 simulation trials** were conducted;
* the planner achieved an **86% success rate**;
* successful runs required approximately **26.4 ± 5.6 s** on average.

These results were intended only as a proof of concept for the proposed planning formulation rather than as a comprehensive benchmark.

## Repository Structure

* `planner.py` — MCTS tree, selection, expansion, evaluation, and online planning logic
* `robot.py` — non-holonomic robot state and discrete action model
* `obstacle.py` — static, moving, and simplified passively deformable obstacle models
* `map.py` — environment simulation, predictive obstacle updates, collision checking, and state evaluation
* `simulate.py` — interactive visualization of a single planning run
* `multiple_simulations.py` — repeated simulation experiments
* `sim_result.json` — saved results from the 50-run experiment

## Running the Demo

The implementation requires Python with:

```bash
pip install numpy matplotlib tqdm
```

Run the interactive demo with:

```bash
python simulate.py
```

To run the repeated simulation experiment:

```bash
python multiple_simulations.py
```

## Limitations

This repository is a small qualification-exam prototype rather than a general-purpose motion planning framework.

In particular:

* the robot is represented by a simplified 2D point model;
* deformable obstacles use a handcrafted low-dimensional interaction model;
* the implemented planner is local and does not include the proposed global-planning layer;
* uncertainty and belief-state estimation were discussed in the broader project but are not implemented here;
* no real-robot experiments were conducted.

The project mainly served to explore whether MCTS could provide a useful framework for **online motion planning when safe interaction with the environment is part of the planning problem**.
