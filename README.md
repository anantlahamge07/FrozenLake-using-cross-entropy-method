# FrozenLake Using Cross-Entropy Method

A learning-focused reinforcement learning project that solves Gymnasium's `FrozenLake-v1` environment using the **Cross-Entropy Method (CEM)** and a small PyTorch policy network.

The project converts FrozenLake's discrete state into a one-hot vector, samples complete episodes from the current policy, keeps the best-performing trajectories, and trains the policy to imitate the actions from those elite episodes.

## What This Project Demonstrates

- Reinforcement learning with the Cross-Entropy Method
- Policy learning with PyTorch
- One-hot encoding for discrete observation spaces
- Elite trajectory filtering based on discounted episode rewards
- TensorBoard logging for training metrics
- A simple, readable training loop for RL beginners

## Environment

The agent trains on:

```text
Gymnasium FrozenLake-v1
```

The current configuration uses:

```python
gym.make("FrozenLake-v1", is_slippery=False)
```

This makes the environment deterministic, which is helpful when learning and debugging the CEM algorithm. In FrozenLake, the agent must move from the start tile to the goal tile while avoiding holes.

## How the Algorithm Works

The Cross-Entropy Method improves a policy by repeatedly learning from the best episodes it has sampled so far.

At a high level:

1. Run the current policy in the environment.
2. Collect a batch of complete episodes.
3. Calculate a discounted score for each episode.
4. Keep only episodes above a reward percentile.
5. Use the states and actions from those elite episodes as supervised training data.
6. Update the policy network with cross-entropy loss.
7. Repeat until the average reward is high enough.

The implementation uses this discounted score:

```text
discounted_reward = episode_reward * GAMMA ^ episode_length
```

This helps prefer shorter successful paths, because reaching the goal in fewer steps receives a stronger discounted score.

## Project Structure

```text
FrozenLake-using-cross-entropy-method/
|-- frozenlake.py             # Main training loop and CEM implementation
|-- non_linear_function.py    # PyTorch policy network
|-- DiscreteOneHotWrapper.py  # Converts discrete states to one-hot vectors
|-- runs/                     # TensorBoard logs
|-- LICENSE
`-- README.md
```

## Main Files

### `frozenlake.py`

Contains the core reinforcement learning logic:

- `EpisodeStep` and `Episode` data classes
- episode generation through `create_batches()`
- elite episode selection through `elite_episodes()`
- PyTorch training loop
- TensorBoard metric logging

### `non_linear_function.py`

Defines the policy network:

```text
Input: one-hot FrozenLake state
      |
Linear(obs_size -> 128)
      |
ReLU
      |
Linear(128 -> number_of_actions)
      |
Action logits
```

The logits are converted into action probabilities with `Softmax` during episode sampling.

### `DiscreteOneHotWrapper.py`

Wraps a Gymnasium environment with a discrete observation space and converts each integer state into a one-hot vector.

FrozenLake normally returns states like:

```text
0, 1, 2, ...
```

The neural network receives vectors like:

```text
[1, 0, 0, 0, ...]
```

## Requirements

- Python 3.10+
- Gymnasium
- NumPy
- PyTorch
- TensorBoard

Install the dependencies with:

```bash
pip install gymnasium numpy torch tensorboard
```

If you are using a GPU or need a specific PyTorch build, install PyTorch using the command recommended by the official PyTorch installation selector.

## Running

From inside the project directory:

```bash
python frozenlake.py
```

During training, the script prints progress like:

```text
iteration: 12, loss = 0.63, reward_mean = 0.42, reward_bound = 0.0
```

Training stops when:

```python
reward_mean > 0.8
```

At that point the script prints:

```text
solved!
```

## Configuration

The main hyperparameters are defined near the top of `frozenlake.py`:

```python
HIDDEN_SIZE = 128
BATCH_SIZE = 100
PERCENTILE = 30
GAMMA = 0.9
```

| Parameter | Meaning | Current value |
|---|---|---:|
| `HIDDEN_SIZE` | Hidden layer size of the policy network | `128` |
| `BATCH_SIZE` | Number of episodes collected per batch | `100` |
| `PERCENTILE` | Reward percentile used to choose elite episodes | `30` |
| `GAMMA` | Discount factor used when ranking episodes | `0.9` |

The optimizer is Adam with a learning rate of `0.001`.

## TensorBoard

Training metrics are written to the `runs/` directory.

Start TensorBoard with:

```bash
tensorboard --logdir runs
```

The script logs:

- `loss`
- `reward_bound`
- `reward mean`

These metrics help track whether the policy is improving over time.

## Notes

- The environment is deterministic because `is_slippery=False`.
- The project is intentionally small and educational.
- The current implementation does not save the trained model.
- For visual debugging, you can experiment with Gymnasium render modes, but the default script trains without rendering.

## License

This project is licensed under the MIT License.
