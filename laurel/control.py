"""Implementations of middle-mile controllers."""

from collections.abc import Mapping, Sequence
import functools
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import jax_dataloader as jdl
import jraph
import numpy as np
import optax

from . import graph_utils
from . import mdp


class LinearSupervisedController:
  """Controller learnt with supervised learning using linear features."""

  def __init__(
      self,
      rng: np.random.Generator,
      key: jax.random.PRNGKeyArray,
      env: mdp.MiddleMileMDP,
      num_rollouts: int,
      num_epochs: int,
      batch_size: int = 32,
      step_prune: bool = False,
      feature_graph_kwargs: Mapping[str, Any] | None = None,
  ):
    self._rng = rng
    self._key = key
    self._env = env
    self._num_rollouts = num_rollouts
    self._num_epochs = num_epochs
    self._batch_size = batch_size
    self._step_prune = step_prune
    self._feature_graph_kwargs = {
      'min_phantom_weight': None, 'prune_parcel': False
    } | dict(feature_graph_kwargs or {})

    self._params = None
    self._policy = nn.Dense(1)

  def get_features(self, state):
    # Get feature graph.
    state, feature_graph, parcel, trucks = self._env.get_feature_graph(
        1, state, **self._feature_graph_kwargs
    )

    if parcel is None:
      # No parcel left in network.
      return state, None, None, None, None, None

    truck_ids = list(trucks.keys())
    if len(trucks) == 1:
      # Only one truck: no need for features.
      return state, None, feature_graph, parcel, trucks, truck_ids

    # Remove unneeded parts from feature graph.
    nodes = feature_graph.nodes[:, 2:]  # Remove node location and time.
    edges = feature_graph.edges[:, 1:]  # Remove edge ids.

    # Turn feature graph into list of vector features.
    edge_features = edges[truck_ids]
    node_features = nodes[feature_graph.receivers[truck_ids]]
    features = np.hstack([edge_features, node_features])

    return state, features, feature_graph, parcel, trucks, truck_ids

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
        (
          state, features, feature_graph, parcel, trucks, truck_ids
        ) = self.get_features(state)
        if parcel is None:
          # No parcel left in network.
          break
        parcel_state, parcel_fg = parcel
        parcel_id = int(
            feature_graph.edges[parcel_fg, graph_utils.EdgeFeatures.ID]
        )

        if len(trucks) == 1:
          # Only one available truck, no need to add to dataset.
          state, *_ = self._env.step(
            parcel_state, trucks[truck_ids[0]], state, prune=self._step_prune
          )
          continue

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

        # Add features and label to dataset.
        dataset.append((features, np.where(truck_ids == label)[0][0]))

        # Use truck and get next state.
        truck = trucks[label]
        state, *_ = self._env.step(
          parcel_state, truck, state, prune=self._step_prune
        )

    return dataset

  @functools.partial(jax.jit, static_argnums=(0,))
  def step(self, params, opt_state, features, mask, labels):
    def risk(params):
      logits = self._policy.apply(params, features)[..., 0]
      logits = jnp.where(mask, logits, -jnp.inf)  # Mask out invalid actions
      loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
      return loss.mean()

    loss, grad = jax.value_and_grad(risk)(params)
    updates, opt_state = self._optimizer.update(grad, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

  def train(self):
    # Collect data and initialize policy.
    dataset = self.collect_data()
    self._key, subkey = jax.random.split(self._key)
    params = self._policy.init(subkey, dataset[0][0][0])

    # Pad features.
    extend_to = max(len(features) for (features, _) in dataset)
    features = np.zeros((len(dataset), extend_to, dataset[0][0].shape[-1]))
    mask = np.zeros((len(dataset), extend_to), bool)
    labels = np.zeros(len(dataset), int)
    for i, (feature, label) in enumerate(dataset):
      features[i, :len(feature)] = feature
      mask[i, :len(feature)] = True
      labels[i] = label
    features = jnp.asarray(features)
    mask = jnp.asarray(mask)
    labels = jnp.asarray(labels)

    # Initialize dataloader and optimizer.
    dataloader = jdl.DataLoaderJax(
      jdl.ArrayDataset(features, mask, labels),
      batch_size=self._batch_size,
      shuffle=True,
      drop_last=True
    )
    schedule = optax.exponential_decay(
      1e-2, self._num_epochs * len(dataloader), 1e-2
    )
    self._optimizer = optax.adam(schedule)
    opt_state = self._optimizer.init(params)

    # Optimize parameters.
    losses = np.zeros(self._num_epochs)
    for i in range(self._num_epochs):
      for (features_, mask_, labels_) in dataloader:
        params, opt_state, loss = self.step(
          params, opt_state, features_, mask_, labels_
        )
        losses[i] += loss / len(dataloader)

    self._params = params
    return losses

  @functools.partial(jax.jit, static_argnums=(0,))
  @functools.partial(jax.vmap, in_axes=(None, None, 0))
  def policy(self, params, features):
    return self._policy.apply(params, features)[..., 0]

  def act(self, state):
    state, features, _, parcel, trucks, truck_ids = self.get_features(state)
    if parcel is None:
      return state, None, None
    if len(trucks) == 1:
      return state, parcel[0], trucks[truck_ids[0]]
    logits = self.policy(self._params, features)
    truck = trucks[truck_ids[np.argmax(logits)]]
    return state, parcel[0], truck


class GNNSupervisedController:
  """Controller learnt with supervised learning using GNN processing."""

  def __init__(
      self,
      rng: np.random.Generator,
      key: jax.random.PRNGKeyArray,
      env: mdp.MiddleMileMDP,
      num_rollouts: int,
      num_epochs: int,
      num_feature_graph_steps: int,
      batch_size: int = 32,
      step_prune: bool = False,
      feature_graph_kwargs: Mapping[str, Any] | None = None,
  ):
    self._rng = rng
    self._key = key
    self._env = env
    self._num_rollouts = num_rollouts
    self._num_epochs = num_epochs
    self._num_feature_graph_steps = num_feature_graph_steps
    self._batch_size = batch_size
    self._step_prune = step_prune
    self._feature_graph_kwargs = {
      'min_phantom_weight': None, 'prune_parcel': False
    } | dict(feature_graph_kwargs or {})

    self._params = None
    self._policy = GraphNet(2 * self._num_feature_graph_steps)
    self._edges_len = None

  def get_features(self, state):
    # Get feature graph.
    state, feature_graph, parcel, trucks = self._env.get_feature_graph(
        self._num_feature_graph_steps, state, **self._feature_graph_kwargs
    )

    if parcel is None:
      # No parcel left in network.
      return state, None, None, None, None, None

    truck_ids = list(trucks.keys())
    if len(trucks) == 1:
      # Only one truck: no need for features.
      return state, None, feature_graph, parcel, trucks, truck_ids

    # Remove unneeded parts from feature graph, add available truck and parcel
    # identifying features.
    nodes = feature_graph.nodes[:, 2:]  # Remove node location and time.
    edges = np.hstack([
        feature_graph.edges[:, 1:],  # Remove edge ids.
        np.isin(np.arange(len(feature_graph.edges)), truck_ids)[:, None],
        (np.arange(len(feature_graph.edges)) == parcel[1])[:, None]
    ])
    features = graph_utils.Graph(
        nodes, edges, feature_graph.receivers, feature_graph.senders
    ).to_jraph()

    return state, features, feature_graph, parcel, trucks, truck_ids

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
        state, features, feature_graph, parcel, trucks, truck_ids = (
          self.get_features(state)
        )
        if parcel is None:
          # No parcel left in network.
          break
        parcel_state, parcel_fg = parcel
        parcel_id = int(
            feature_graph.edges[parcel_fg, graph_utils.EdgeFeatures.ID]
        )

        if len(trucks) == 1:
          # Only one available truck, no need to add to dataset.
          state, *_ = self._env.step(
            parcel_state, trucks[truck_ids[0]], state, prune=self._step_prune
          )
          continue

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

        # Add features and label to dataset.
        dataset.append((features, truck_ids, parcel_fg, label))

        # Use truck and get next state.
        truck = trucks[label]
        state, *_ = self._env.step(
          parcel_state, truck, state, prune=self._step_prune
        )

    return dataset

  @functools.partial(jax.jit, static_argnums=(0,))
  def step(self, params, opt_state, features, labels):
    def risk(params):
      # Apply GNN.
      graph = self._policy.apply(params, features)

      # Mask out invalid actions, split by individual graphs.
      logits = jnp.where(features.edges[:, -2], graph.edges.ravel(), -jnp.inf)
      logits = logits.reshape(-1, self._edges_len)

      # Compute cross entropy loss.
      loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
      return loss.mean()

    loss, grad = jax.value_and_grad(risk)(params)
    updates, opt_state = self._optimizer.update(grad, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

  def train(self):
    # Collect data and pad to make all graphs equal-sized.
    dataset = self.collect_data()
    sizes = np.array(
      [(len(graph.nodes) + 1, len(graph.edges)) for graph, *_ in dataset]
    ).max(0)
    self._edges_len = sizes[1]
    dataset = [
      (jraph.pad_with_graphs(graph, *sizes), *rest) for graph, *rest in dataset
    ]

    # Initialize policy.
    self._key, subkey = jax.random.split(self._key)
    params = self._policy.init(subkey, dataset[0][0])

    # Initialize optimizer.
    schedule = optax.exponential_decay(
      1e-2, self._num_epochs * (len(dataset) // self._batch_size), 1e-2
    )
    self._optimizer = optax.adam(schedule)
    opt_state = self._optimizer.init(params)

    # Optimize parameters.
    losses = np.zeros(self._num_epochs)
    for i in range(self._num_epochs):
      length = (len(dataset) // self._batch_size) * self._batch_size

      # Sample dataset permutation, iterate over batches.
      idx = self._rng.permutation(len(dataset))
      idx = idx[:length]
      idx = idx.reshape(-1, self._batch_size)
      for batch_idx in idx:
        batch = jraph.batch([dataset[i][0] for i in batch_idx])
        labels = jnp.array([dataset[i][-1] for i in batch_idx], int)
        params, opt_state, loss = self.step(params, opt_state, batch, labels)
        losses[i] += loss / length

    self._params = params
    return losses

  @functools.partial(jax.jit, static_argnums=(0,))
  def policy(self, params, features):
    return self._policy.apply(params, features)

  def act(self, state):
    state, features, _, parcel, trucks, truck_ids = self.get_features(state)
    if parcel is None:
      return state, None, None
    if len(trucks) == 1:
      return state, parcel[0], trucks[truck_ids[0]]
    features = graph_utils.pad_graphs_tuple(features)
    logits = self.policy(self._params, features).edges.ravel()[
      jnp.array(truck_ids)
    ]
    truck = trucks[truck_ids[np.argmax(logits)]]
    return state, parcel[0], truck


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
    )
    processor = jraph.GraphNetwork(
        update_edge_fn=jraph_mlp([16, 16]),
        update_node_fn=jraph_mlp([16, 16]),
    )
    decoder = jraph.GraphNetwork(
        update_edge_fn=jraph_mlp([16, 1]),
        update_node_fn=None,
    )

    x = encoder(state)
    for _ in range(self.process_steps):
      x = processor(x)
    x = decoder(x)
    return x


class MLP(nn.Module):
  layers: Sequence[int]

  @nn.compact
  def __call__(self, x: jax.Array):
    for features in self.layers[:-1]:
      x = nn.Dense(features)(x)
      x = nn.silu(x)
    x = nn.Dense(self.layers[-1])(x)
    return x
