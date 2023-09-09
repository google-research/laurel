"""Implementations of middle-mile controllers."""
# TODO(onno): This module is not ready yet!

from collections.abc import Mapping, Sequence
import functools
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import jraph
import numpy as np
from scipy import special

from google3.util.operations_research.cloud.lattle.experiments import graph_utils
from google3.util.operations_research.cloud.lattle.experiments import mdp


class SupervisedController:
  """Controller learnt with supervised learning."""

  def __init__(
      self,
      rng: np.random.Generator,
      env: mdp.MiddleMileMDP,
      num_rollouts: int,
      num_training_steps: int,
      num_feature_graph_steps: int,
      linear_features: bool,
      feature_graph_kwargs: Mapping[str, Any] | None = None,
  ):
    self._rng = rng
    self._env = env
    self._num_rollouts = num_rollouts
    self._num_training_steps = num_training_steps
    self._num_feature_graph_steps = num_feature_graph_steps
    self._linear_features = linear_features
    self._feature_graph_kwargs = feature_graph_kwargs or {}

    if self._linear_features:
      self.policy = nn.Dense(1)
      # TODO(onno): Continue here :)

  def collect_data(
      self,
  ) -> Sequence[
      tuple[Mapping[int, jax.Array], int]
      | tuple[jraph.GraphsTuple, Sequence[int], int, int]
  ]:
    """Collect dataset by running rollouts."""
    dataset = []
    for _ in range(self._num_rollouts):
      state, solution = self._env.reset(self._rng)
      while True:
        # Get feature graph.
        state, feature_graph, parcel, trucks = self._env.get_feature_graph(
            self._num_feature_graph_steps, state, **self._feature_graph_kwargs
        )
        if parcel is None:
          # No parcel left in network.
          break
        parcel_state, parcel_fg = parcel
        parcel_id = int(
            feature_graph.edges[parcel_fg, graph_utils.EdgeFeatures.ID]
        )

        # Find the correct truck according to the solution.
        label = None
        for truck in trucks:
          sender = tuple(
              feature_graph.nodes[feature_graph.senders[truck], :2].astype(int)
          )
          receiver = tuple(
              feature_graph.nodes[feature_graph.receivers[truck], :2].astype(
                  int
              )
          )
          for first_stop, second_stop in zip(
              solution[parcel_id], solution[parcel_id][1:]
          ):
            if (first_stop, second_stop) == (sender, receiver):
              label = truck
              break
          else:
            continue
          break

        # Remove unneeded parts from feature graph, add to dataset.
        nodes = feature_graph.nodes[:, 2:]  # Remove node location and time.
        edges = feature_graph.edges[:, 1:]  # Remove edge ids.
        feature_graph = graph_utils.Graph(
            nodes, edges, feature_graph.receivers, feature_graph.senders
        )
        if self._linear_features:
          # Turn feature graph into list of vector features.
          edge_features = feature_graph.edges[trucks]
          node_features = feature_graph.nodes[feature_graph.receivers[trucks]]
          features = dict(
              zip(trucks, jnp.hstack([edge_features, node_features]))
          )
          dataset.append((features, label))
        else:
          dataset.append(
              (feature_graph.to_jraph(), list(trucks.keys()), parcel_fg, label)
          )

        # Use truck and get next state.
        truck = trucks[label]
        state, *_ = self._env.step(parcel_state, truck, state)

    return dataset


class LinearPolicy:
  pass


class FVIController:
  """Fitted value iteration control algorithm."""

  def __init__(
      self,
      env: mdp.MiddleMileMDP,
      rng: np.random.Generator,
      key: jax.random.PRNGKeyArray,
  ):
    self.env = env
    self.rng = rng
    self.buffer_idx = 0
    self.buffer_size = 1000
    self.batch_size = 10

    # Initialize environment and replay buffer
    self.state = self.env.reset(self.rng)
    self.buffer = [self.state]

    # Initialize value function.
    self.value_fn = GraphNet(process_steps=10)
    self.value_params = self.value_fn.init(key, self.state.to_jraph())

  def learn(self, num_steps: int):
    # TODO(onno): Repeat collection/update loop.
    pass

  @functools.partial(jax.jit, static_argnums=0)
  def value(self, params: Any, state: Any) -> jax.Array:
    return self.value_fn.apply(params, state)

  def collect_data(self, num_steps: int):
    """Sample transitions from environment and add to replay buffer."""
    for _ in range(num_steps):
      # Get next state.
      action = self.act(self.state, explore=True)
      if action is None:
        self.state = self.env.reset(self.rng)
      else:
        self.state = self.env.step(*action, self.state)

      # Add state to replay buffer.
      self.buffer[self.buffer_idx : (self.buffer_idx + 1)] = [self.state]
      self.buffer_idx = (self.buffer_idx + 1) % self.buffer_size

  # @functools.partial(jax.jit, static_argnums=0)
  # def loss(self, params, states, targets):
  #   values = self.value(params, state)

  # @functools.partial(jax.jit, static_argnums=0)
  # def optimize(self, params, states, targets, opt_state):
  #   # return params, opt_state
  #   pass

  def update(self, num_steps: int):
    """Update the value function using data from the replay buffer."""
    for _ in range(num_steps):
      # Sample a batch from buffer.
      states = self.rng.choice(self.buffer, self.batch_size, replace=False)

      # Compute Bellman targets.
      # TODO(onno): it might be more efficient to store all possible next
      # states in the replay buffer. This depends on the speed of `env.step`.
      targets = np.zeros(self.batch_size)
      for i, state in enumerate(states):
        actions = self.env.get_actions(state)
        if actions is not None:
          parcel, trucks = actions
          for truck in trucks:
            next_state = self.env.step(parcel, truck, state).to_jraph(
                padded=True
            )  # TODO(onno): No reward yet. (use delivery!) -> actually, all
            # env.steps here are wrong because they ignore the delivery variable
            targets[i] = max(self.value(self.params, next_state)[0], targets[i])

      # Perform one optimization step.
      states = graph_utils.pad_graphs_tuple(
          jraph.batch([graph.to_jraph() for graph in states])
      )
      targets = jnp.asarray(targets)
      self.params, self.opt_state = self.optimize(
          self.params, states, targets, self.opt_state
      )

  def act(
      self, state: graph_utils.Graph, explore: bool = False
  ) -> tuple[int, int]:
    """Policy based on learned value function and Boltzmann exploration."""
    # Get available actions (parcel with trucks) from given state.
    actions = self.env.get_actions(state)
    if actions is None:
      return None
    parcel, trucks = actions

    # No need to compute values for single available action.
    if len(trucks) == 1:
      return parcel, trucks[0]

    # Calculate values of resulting next states for all actions.
    values = np.zeros(len(trucks))
    for i, truck in enumerate(trucks):
      next_state = self.env.step(parcel, truck, state)
      values[i] = self.value(
          self.value_params, next_state.to_jraph(padded=True)
      )[0]

    if explore:
      # Sample action from Boltzmann distribution.
      probabilities = special.softmax(self.inv_temp * values)
      index = np.where(self.rng.multinomial(1, probabilities))[0][0]
      return parcel, trucks[index]

    # Return best action without exploration.
    return parcel, trucks[np.argmax(values)]


class GraphNet(nn.Module):
  """Encode-process-decode graph net architecture."""

  process_steps: int

  @nn.compact
  def __call__(self, state: jraph.GraphsTuple):
    def jraph_mlp(layers):
      return jraph.concatenated_args(MLP(layers))

    encoder = jraph.GraphNetwork(
        update_edge_fn=jraph_mlp([16, 16]),
        update_node_fn=jraph_mlp([16, 16]),
        update_global_fn=jraph_mlp([8, 8]),
    )
    processor = jraph.GraphNetwork(
        update_edge_fn=jraph_mlp([16, 16]),
        update_node_fn=jraph_mlp([16, 16]),
        update_global_fn=jraph_mlp([8, 8]),
    )
    decoder = jraph.GraphNetwork(
        update_edge_fn=jraph_mlp([16, 16]),
        update_node_fn=jraph_mlp([16, 16]),
        update_global_fn=jraph_mlp([8, 1]),
    )

    x = state._replace(globals=jnp.zeros([state.n_node.shape[0], 1]))
    x = encoder(x)
    for _ in range(self.process_steps):
      x = processor(x)
    x = decoder(x)
    return x.globals[:, 0]


class MLP(nn.Module):
  layers: Sequence[int]

  @nn.compact
  def __call__(self, x: jax.Array):
    for features in self.layers[:-1]:
      x = nn.Dense(features)(x)
      x = nn.silu(x)
    x = nn.Dense(self.layers[-1])(x)
    return x
