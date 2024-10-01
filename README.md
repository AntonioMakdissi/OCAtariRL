OCAtariRL: Playing Object-Centric Atari Games with Reinforcement Learning
=========================================================================

This project explores object-centric reinforcement learning (RL) environments using Atari 2600 games. It focuses on the interpretability and consistency of state representations, allowing RL agents to make decisions based on object-like inputs rather than pixel-based observations.

Project Overview
----------------
This project aims to enhance both interpretability and performance of RL models by:
- Using object-centric observations instead of raw pixels.
- Ensuring consistent object observations across different states.
- Reducing training complexity through structured object representations.

The project uses Proximal Policy Optimization (PPO) as the RL algorithm and benchmarks its performance on various Atari games.

Key Features
------------
- **Object-Centric Atari Environment**: Built using the OCAtari framework, transforming pixel inputs into object-based representations.
- **Interpretable RL**: The project improves the interpretability of RL models by providing a clearer link between observations and agent actions.
- **Efficient Training**: By focusing on objects, the training is expected to be faster and more efficient compared to pixel-based inputs.


Results
-------
The study shows that object-centric RL agents can generalize better in normalized environments but pixel-based approaches still outperform in some scenarios. See the `results/` directory for detailed benchmarks and comparisons.

Future Work
-----------
- Investigating hybrid approaches combining pixel and object-centric observations.
- Exploring more efficient methods for object detection and state consistency.

