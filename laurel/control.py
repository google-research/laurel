"""Implementations of middle-mile controllers."""

from collections.abc import Mapping, Sequence
import functools
from typing import Any, Optional, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
import jax_dataloader as jdl
import jraph
import numpy as np
import optax
from scipy import special

from laurel import graph_utils
from laurel import mdp


class LinearSupervisedController:
  """Controller learnt with supervised learning using linear features."""

  def __init__(
      self,
      rng: np.random.Generator,
      key: jax.Array,
      env: mdp.MiddleMileMDP,
      num_rollouts: int,
      num_epochs: int,
      batch_size: int = 32,
      step_prune: bool = False,
      feature_graph_kwargs: Optional[Mapping[str, Any]] = None,
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
      pb=None
  ) -> Sequence[Union[
      tuple[Mapping[int, jax.Array], int],
      tuple[jraph.GraphsTuple, Sequence[int], int, int]
  ]]:
    """Collect dataset by running rollouts."""
    pb = pb if pb is not None else (lambda x: x)
    dataset = []
    for _ in pb(range(self._num_rollouts)):
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
          for i, first_stop in enumerate(solution[parcel_id]):
            if first_stop == sender:
              for second_stop in solution[parcel_id][i + 1:]:
                if second_stop == receiver:
                  label = truck
                  break
              else:
                continue
              break

        if label is None:
          # Truck not found (something went wrong). Put parcel onto random truck
          # and continue.
          truck = trucks[self._rng.choice(truck_ids)]
          state, *_ = self._env.step(
            parcel_state, truck, state, prune=self._step_prune
          )
          continue

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

  def train(self, pb_data=None, pb_epoch=None, pb_batch=None):
    pb_epoch = pb_epoch if pb_epoch is not None else (lambda x: x)
    pb_batch = pb_batch if pb_batch is not None else (lambda x: x)

    # Collect data and initialize policy.
    dataset = self.collect_data(pb=pb_data)
    self._key, subkey = jax.random.split(self._key)
    self._params = self._policy.init(subkey, dataset[0][0][0])

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
    opt_state = self._optimizer.init(self._params)

    # Optimize parameters.
    losses = np.zeros(self._num_epochs)
    for i in pb_epoch(range(self._num_epochs)):
      for (features_, mask_, labels_) in pb_batch(dataloader):
        self._params, opt_state, loss = self.step(
          self._params, opt_state, features_, mask_, labels_
        )
        losses[i] += loss / len(dataloader)

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
      key: jax.Array,
      env: mdp.MiddleMileMDP,
      num_rollouts: int,
      num_epochs: int,
      num_feature_graph_steps: int,
      batch_size: int = 32,
      step_prune: bool = False,
      feature_graph_kwargs: Optional[Mapping[str, Any]] = None,
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
      pb=None
  ) -> Sequence[Union[
      tuple[Mapping[int, jax.Array], int],
      tuple[jraph.GraphsTuple, Sequence[int], int, int]
  ]]:
    """Collect dataset by running rollouts."""
    pb = pb if pb is not None else (lambda x: x)
    dataset = []
    for _ in pb(range(self._num_rollouts)):
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
          for i, first_stop in enumerate(solution[parcel_id]):
            if first_stop == sender:
              for second_stop in solution[parcel_id][i + 1:]:
                if second_stop == receiver:
                  label = truck
                  break
              else:
                continue
              break

        if label is None:
          # Truck not found (something went wrong). Put parcel onto random truck
          # and continue.
          truck = trucks[self._rng.choice(truck_ids)]
          state, *_ = self._env.step(
            parcel_state, truck, state, prune=self._step_prune
          )
          continue

        # Add features and label to dataset.
        dataset.append((features, truck_ids, parcel_fg, label))

        # Use truck and get next state.
        truck = trucks[label]
        state, *_ = self._env.step(
          parcel_state, truck, state, prune=self._step_prune
        )

    return dataset

  @functools.partial(jax.jit, static_argnums=(0,))
  def loss(self, params, features, labels):
    # Apply GNN.
    graph = self._policy.apply(params, features)

    # Mask out invalid actions, split by individual graphs.
    logits = jnp.where(features.edges[:, -2], graph.edges.ravel(), -jnp.inf)
    logits = logits.reshape(-1, self._edges_len)

    # Compute cross entropy loss.
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
    return loss.mean()

  @functools.partial(jax.jit, static_argnums=(0,))
  def step(self, params, opt_state, features, labels):
    loss, grad = jax.value_and_grad(self.loss)(params, features, labels)
    updates, opt_state = self._optimizer.update(grad, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

  def train(self, pb_data=None, pb_epoch=None, pb_batch=None):
    pb_epoch = pb_epoch if pb_epoch is not None else (lambda x: x)
    pb_batch = pb_batch if pb_batch is not None else (lambda x: x)

    # Collect data and pad to make all graphs equal-sized.
    dataset = self.collect_data(pb=pb_data)
    sizes = np.array(
      [(len(graph.nodes) + 1, len(graph.edges)) for graph, *_ in dataset]
    ).max(0)
    self._edges_len = sizes[1]
    dataset = [
      (jraph.pad_with_graphs(graph, *sizes), *rest) for graph, *rest in dataset
    ]

    # Initialize policy.
    self._key, subkey = jax.random.split(self._key)
    self._params = self._policy.init(subkey, dataset[0][0])

    # Initialize optimizer.
    schedule = optax.exponential_decay(
      1e-2, self._num_epochs * (len(dataset) // self._batch_size), 1e-2
    )
    self._optimizer = optax.adam(schedule)
    opt_state = self._optimizer.init(self._params)

    # Optimize parameters.
    losses = np.zeros(self._num_epochs)
    for i in pb_epoch(range(self._num_epochs)):
      length = (len(dataset) // self._batch_size) * self._batch_size

      # Sample dataset permutation, iterate over batches.
      idx = self._rng.permutation(len(dataset))
      idx = idx[:length]
      idx = idx.reshape(-1, self._batch_size)
      for batch_idx in pb_batch(idx):
        # Training step.
        batch = jraph.batch([dataset[i][0] for i in batch_idx])
        labels = jnp.array([dataset[i][-1] for i in batch_idx], int)
        self._params, opt_state, loss = self.step(
          self._params, opt_state, batch, labels
        )
        losses[i] += loss / len(idx)

    return losses

  @functools.partial(jax.jit, static_argnums=(0,))
  def policy(self, params, features):
    return self._policy.apply(params, features).edges.ravel()

  def act(self, state):
    state, features, _, parcel, trucks, truck_ids = self.get_features(state)
    if parcel is None:
      return state, None, None
    if len(trucks) == 1:
      return state, parcel[0], trucks[truck_ids[0]]
    features = graph_utils.pad_graphs_tuple(features)
    logits = self.policy(self._params, features)[jnp.array(truck_ids)]
    truck = trucks[truck_ids[np.argmax(logits)]]
    return state, parcel[0], truck


class LinearPPO:
  """Proximal policy optimization controller using a linear policy."""

  def __init__(
      self,
      rng: np.random.Generator,
      key: jax.Array,
      env: mdp.MiddleMileMDP,
      num_rollouts_total: int,
      num_rollouts: int,
      num_actor_updates: int,
      num_critic_updates: int,
      actor_lr: float,
      critic_lr: float,
      exploration_inv_temperature: float = 0.1,
      clip_eps: float = 0.2,
      kl_stop: float = 0.1,
      batch_size: int = 32,
      step_prune: bool = False,
      feature_graph_kwargs: Optional[Mapping[str, Any]] = None,
  ):
    self._rng = rng
    self._key = key
    self._env = env
    self._num_epochs = num_rollouts_total // num_rollouts
    self._num_rollouts = num_rollouts
    self._num_actor_updates = num_actor_updates
    self._num_critic_updates = num_critic_updates
    self._actor_lr = actor_lr
    self._critic_lr = critic_lr
    self._exploration_inv_temperature = exploration_inv_temperature
    self._clip_eps = clip_eps
    self._kl_stop = kl_stop
    self._batch_size = batch_size
    self._step_prune = step_prune
    self._feature_graph_kwargs = {
      'min_phantom_weight': None, 'prune_parcel': False
    } | dict(feature_graph_kwargs or {})

    self._actor = nn.Dense(1)
    self._critic = nn.Dense(1)

  def get_features(self, state):
    # Get feature graph.
    state, feature_graph, parcel, trucks = self._env.get_feature_graph(
        1, state, **self._feature_graph_kwargs
    )
    if parcel is None:
      # No parcel left in network.
      return state, None, None, None, None, None
    truck_ids = list(trucks.keys())

    # Remove unneeded parts from feature graph.
    nodes = feature_graph.nodes[:, 2:]  # Remove node location and time.
    edges = feature_graph.edges[:, 1:]  # Remove edge ids.

    # Turn feature graph into list of vector features.
    edge_features = edges[truck_ids]
    node_features = nodes[feature_graph.receivers[truck_ids]]
    features = np.hstack([edge_features, node_features])

    return state, features, feature_graph, parcel, trucks, truck_ids

  def rollout(self):
    observations = []
    actions = []
    rewards = []
    logps = []
    exploration = 0
    deliveries = 0
    state, _ = self._env.reset(self._rng)
    while True:
      state, features, _, parcel, trucks, truck_ids = (
        self.get_features(state)
      )
      if parcel is None:
        # No parcel left in network.
        break
      if len(trucks) == 1:
        # Only one available truck, no need to add to dataset.
        state, delivery, _ = self._env.step(
          parcel[0], trucks[truck_ids[0]], state, prune=self._step_prune
        )
        deliveries += delivery
        if delivery and rewards:
          # Let's say this delivery is a result of the previous action.
          rewards[-1] += 1
        continue

      # Use truck and get next state.
      logits = special.log_softmax(self.policy(self._actor_params, features))
      action = self._rng.choice(len(trucks), p=np.exp(logits))
      exploration += action == np.argmax(logits)
      truck = trucks[truck_ids[action]]
      state, delivery, _ = self._env.step(
        parcel[0], truck, state, prune=self._step_prune
      )
      deliveries += delivery

      # Add transition to dataset.
      observations.append(features)
      actions.append(action)
      rewards.append(int(delivery))
      logps.append(logits)

    return (
      observations,
      actions,
      rewards,
      logps,
      exploration / len(logps),
      deliveries / self._env._num_parcels
    )

  @functools.partial(jax.vmap, in_axes=(None, None, None, 0, 0, 0, 0))
  @functools.partial(jax.jit, static_argnums=(0,))
  def actor_loss(
    self, actor_params, critic_params, observation, action, logps_old, mask
  ):
    # Compute advantage estimate.
    q_values = self.q(critic_params, observation)
    value = q_values @ jnp.exp(logps_old)
    advantage = q_values[action] - value

    # Compute PPO loss.
    logps = self.policy(actor_params, observation)
    logps = nn.log_softmax(jnp.where(mask, logps, -jnp.inf))
    ratio = jnp.exp(logps[action] - logps_old[action])
    loss = -jnp.minimum(
      ratio * advantage,
      jnp.clip(ratio, 1 - self._clip_eps, 1 + self._clip_eps) * advantage
    )
    kl = logps_old[action] - logps[action]
    return loss, kl

  @functools.partial(jax.jit, static_argnums=(0,))
  def actor_update(
      self,
      params,
      opt_state,
      critic_params,
      observations,
      actions,
      logps_old,
      mask
  ):
    def risk(params):
      losses, kls = self.actor_loss(
        params, critic_params, observations, actions, logps_old, mask
      )
      return losses.mean(), kls.mean()
    (loss, kl), grad = jax.value_and_grad(risk, has_aux=True)(params)
    updates, opt_state = self._actor_optimizer.update(grad, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss, kl

  @functools.partial(jax.vmap, in_axes=(None, None, 0, 0, 0))
  @functools.partial(jax.jit, static_argnums=(0,))
  def critic_loss(
    self, critic_params, observation, action, return_
  ):
    q_value = self._critic.apply(critic_params, observation[action])
    return (q_value - return_) ** 2

  @functools.partial(jax.jit, static_argnums=(0,))
  def critic_update(
    self, params, opt_state, observations, actions, returns
  ):
    def risk(params):
      loss = self.critic_loss(params, observations, actions, returns)
      return loss.mean()
    loss, grad = jax.value_and_grad(risk)(params)
    updates, opt_state = self._critic_optimizer.update(grad, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

  def train(
      self, pb_epoch=None, pb_rollout=None, pb_actor=None, pb_critic=None
  ):
    pb_epoch = pb_epoch if pb_epoch is not None else (lambda x: x)
    pb_rollout = pb_rollout if pb_rollout is not None else (lambda x: x)
    pb_actor = pb_actor if pb_actor is not None else (lambda x: x)
    pb_critic = pb_critic if pb_critic is not None else (lambda x: x)

    # Initialize environment, actor, critic.
    state, _ = self._env.reset(self._rng)
    state, features, *_ = self.get_features(state)
    self._key, key1, key2 = jax.random.split(self._key, 3)
    self._actor_params = self._actor.init(key1, features[0])
    self._critic_params = self._critic.init(key2, features[0])

    # Initialize optimizers.
    self._actor_optimizer = optax.adam(self._actor_lr)
    self._critic_optimizer = optax.adam(self._critic_lr)
    self._actor_opt_state = self._actor_optimizer.init(self._actor_params)
    self._critic_opt_state = self._critic_optimizer.init(self._critic_params)

    # Main training loop.
    actor_losses = np.zeros((self._num_epochs, self._num_actor_updates))
    critic_losses = np.zeros((self._num_epochs, self._num_critic_updates))
    exploration = np.zeros((self._num_epochs, self._num_rollouts))
    performance = np.zeros((self._num_epochs, self._num_rollouts))
    for i in pb_epoch(range(self._num_epochs)):
      # Collect trajectories in environment.
      observations_het = []  # Heterogeneous data.
      actions = []
      rewards = []
      returns = []
      logps_het = []  # Heterogeneous data.
      max_num_actions = 0
      for j in pb_rollout(range(self._num_rollouts)):
        os, as_, rs, lps, exp, perf = self.rollout()
        observations_het.extend(os)
        actions.extend(as_)
        rewards.extend(rs)
        returns.extend(np.cumsum(rs[::-1])[::-1])
        logps_het.extend(lps)
        max_num_actions = max([max_num_actions] + [len(o) for o in os])
        exploration[i, j] = exp
        performance[i, j] = perf

      # Round padding size to next power of 2.
      max_num_actions = int(2 ** np.ceil(np.log2(max_num_actions)))

      # Pad heteregeneous data.
      num_steps = len(observations_het)
      observations = np.zeros(
        (num_steps, max_num_actions, observations_het[0].shape[-1])
      )
      logps = np.ones((num_steps, max_num_actions)) * (-np.inf)
      mask = np.zeros((num_steps, max_num_actions), bool)
      for t in range(num_steps):
        observations[t, :len(observations_het[t])] = observations_het[t]
        logps[t, :len(observations_het[t])] = logps_het[t]
        mask[t, :len(observations_het[t])] = True

      # Update the policy.
      for k in pb_actor(range(self._num_actor_updates)):
        params, opt_state, loss, kl = self.actor_update(
          self._actor_params,
          self._actor_opt_state,
          self._critic_params,
          jnp.asarray(observations),
          jnp.asarray(actions),
          jnp.asarray(logps),
          jnp.asarray(mask)
        )
        if kl > self._kl_stop:
          break  # Early stopping if KL divergence is too large.
        self._actor_params = params
        self._actor_opt_state = opt_state
        actor_losses[i, k] = loss

      # Update the critic.
      for k in pb_critic(range(self._num_critic_updates)):
        params, opt_state, loss = self.critic_update(
          self._critic_params,
          self._critic_opt_state,
          jnp.asarray(observations),
          jnp.asarray(actions),
          jnp.asarray(returns)
        )
        self._critic_params = params
        self._critic_opt_state = opt_state
        critic_losses[i, k] = loss

    return actor_losses, critic_losses, exploration, performance

  @functools.partial(jax.jit, static_argnums=(0,))
  @functools.partial(jax.vmap, in_axes=(None, None, 0))
  def q(self, params, features):
    return self._critic.apply(params, features)[..., 0]

  @functools.partial(jax.jit, static_argnums=(0,))
  @functools.partial(jax.vmap, in_axes=(None, None, 0))
  def policy(self, params, features):
    return (
      self._actor.apply(params, features)[..., 0]
      * self._exploration_inv_temperature
    )

  def act(self, state):
    state, features, _, parcel, trucks, truck_ids = self.get_features(state)
    if parcel is None:
      return state, None, None
    if len(trucks) == 1:
      return state, parcel[0], trucks[truck_ids[0]]
    logits = self.policy(self._actor_params, features)
    truck = trucks[truck_ids[np.argmax(logits)]]
    return state, parcel[0], truck


class GNN_PPO:
  """Proximal policy optimization controller using a GNN policy."""

  def __init__(
      self,
      rng: np.random.Generator,
      key: jax.Array,
      env: mdp.MiddleMileMDP,
      num_rollouts_total: int,
      num_rollouts: int,
      num_actor_updates: int,
      num_critic_updates: int,
      actor_lr: float,
      critic_lr: float,
      num_feature_graph_steps: int,
      exploration_inv_temperature: float = 0.1,
      clip_eps: float = 0.2,
      kl_stop: float = 0.1,
      batch_size: int = 32,
      step_prune: bool = False,
      feature_graph_kwargs: Optional[Mapping[str, Any]] = None,
  ):
    self._rng = rng
    self._key = key
    self._env = env
    self._num_epochs = num_rollouts_total // num_rollouts
    self._num_rollouts = num_rollouts
    self._num_actor_updates = num_actor_updates
    self._num_critic_updates = num_critic_updates
    self._actor_lr = actor_lr
    self._critic_lr = critic_lr
    self._exploration_inv_temperature = exploration_inv_temperature
    self._clip_eps = clip_eps
    self._kl_stop = kl_stop
    self._num_feature_graph_steps = num_feature_graph_steps
    self._batch_size = batch_size
    self._step_prune = step_prune
    self._feature_graph_kwargs = {
      'min_phantom_weight': None, 'prune_parcel': False
    } | dict(feature_graph_kwargs or {})

    self._actor = GraphNet(2 * self._num_feature_graph_steps)
    self._critic = GraphNet(2 * self._num_feature_graph_steps)

  def get_features(self, state):
    # Get feature graph.
    state, feature_graph, parcel, trucks = self._env.get_feature_graph(
        self._num_feature_graph_steps, state, **self._feature_graph_kwargs
    )
    if parcel is None:
      # No parcel left in network.
      return state, None, None, None, None, None
    truck_ids = list(trucks.keys())

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

  def rollout(self):
    observations = []
    actions = []
    rewards = []
    logps = []
    exploration = 0
    deliveries = 0
    state, _ = self._env.reset(self._rng)
    while True:
      state, features, _, parcel, trucks, truck_ids = (
        self.get_features(state)
      )
      if parcel is None:
        # No parcel left in network.
        break
      if len(trucks) == 1:
        # Only one available truck, no need to add to dataset.
        state, delivery, _ = self._env.step(
          parcel[0], trucks[truck_ids[0]], state, prune=self._step_prune
        )
        deliveries += delivery
        if delivery and rewards:
          # Let's say this delivery is a result of the previous action.
          rewards[-1] += 1
        continue

      # Use truck and get next state.
      features_padded = graph_utils.pad_graphs_tuple(features)
      logits = self.policy(self._actor_params, features_padded)
      logits = jnp.where(features_padded.edges[:, -2], logits, -jnp.inf)
      logits = special.log_softmax(logits)
      # logits = special.log_softmax(
      #   self.policy(self._actor_params, features_padded)[jnp.array(truck_ids)]
      # )
      action = self._rng.choice(len(logits), p=np.exp(logits))
      exploration += action != np.argmax(logits)
      truck = trucks[action]
      state, delivery, _ = self._env.step(
        parcel[0], truck, state, prune=self._step_prune
      )
      deliveries += delivery

      # Add transition to dataset.
      observations.append(features)
      actions.append(action)
      rewards.append(int(delivery))
      logps.append(logits)

    return (
      observations,
      actions,
      rewards,
      logps,
      exploration / len(logps),
      deliveries / self._env._num_parcels
    )

  @functools.partial(jax.vmap, in_axes=(None, 0, 0, 0, 0, 0, 0))
  @functools.partial(jax.jit, static_argnums=(0,))
  def actor_loss(self, q_values, logps, action, logps_old, trucks, mask):
    # Compute advantage estimate.
    value = q_values @ jnp.exp(logps_old)
    advantage = q_values[action] - value

    # Compute PPO loss.
    logps = nn.log_softmax(jnp.where(trucks, logps, -jnp.inf))
    # logps = nn.log_softmax(logps[trucks])
    ratio = jnp.exp(logps[action] - logps_old[action])
    loss = -jnp.minimum(
      ratio * advantage,
      jnp.clip(ratio, 1 - self._clip_eps, 1 + self._clip_eps) * advantage
    )
    kl = logps_old[action] - logps[action]
    return loss, kl

  @functools.partial(jax.jit, static_argnums=(0, 8))
  def actor_update(
      self,
      params,
      opt_state,
      critic_params,
      observations,
      actions,
      logps_old,
      mask,
      num_edges
  ):
    def risk(params):
      # Apply actor and critic GNNs.
      q_values = self.q(critic_params, observations).reshape(-1, num_edges)
      logps = self.policy(params, observations).reshape(-1, num_edges)
      trucks = (observations.edges[:, -2] == 1).reshape(-1, num_edges)
      losses, kls = self.actor_loss(
        q_values, logps, actions, logps_old, trucks, mask
      )
      return losses.mean(), kls.mean()
    (loss, kl), grad = jax.value_and_grad(risk, has_aux=True)(params)
    updates, opt_state = self._actor_optimizer.update(grad, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss, kl

  @functools.partial(jax.vmap, in_axes=(None, 0, 0, 0))
  @functools.partial(jax.jit, static_argnums=(0,))
  def critic_loss(self, q_values, action, return_):
    q_value = q_values[action]
    return (q_value - return_) ** 2

  @functools.partial(jax.jit, static_argnums=(0, 6))
  def critic_update(
    self, params, opt_state, observations, actions, returns, num_edges
  ):
    def risk(params):
      q_values = self.q(params, observations).reshape(-1, num_edges)
      # trucks = (observations.edges[:, -2] == 1).reshape(-1, num_edges)
      loss = self.critic_loss(q_values, actions, returns)
      return loss.mean()
    loss, grad = jax.value_and_grad(risk)(params)
    updates, opt_state = self._critic_optimizer.update(grad, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

  def train(
      self, pb_epoch=None, pb_rollout=None, pb_actor=None, pb_critic=None
  ):
    pba = pb_actor is not None
    pb_epoch = pb_epoch if pb_epoch is not None else (lambda x: x)
    pb_rollout = pb_rollout if pb_rollout is not None else (lambda x: x)
    pb_actor = pb_actor if pb_actor is not None else (lambda x: x)
    pb_critic = pb_critic if pb_critic is not None else (lambda x: x)

    # Initialize environment, actor, critic.
    state, _ = self._env.reset(self._rng)
    state, features, *_ = self.get_features(state)
    self._key, key1, key2 = jax.random.split(self._key, 3)
    self._actor_params = self._actor.init(key1, features)
    self._critic_params = self._critic.init(key2, features)

    # Initialize optimizers.
    self._actor_optimizer = optax.adam(self._actor_lr)
    self._critic_optimizer = optax.adam(self._critic_lr)
    self._actor_opt_state = self._actor_optimizer.init(self._actor_params)
    self._critic_opt_state = self._critic_optimizer.init(self._critic_params)

    # Main training loop.
    actor_losses_ = [[] for _ in range(self._num_epochs)]
    critic_losses_ = [[] for _ in range(self._num_epochs)]
    exploration = np.zeros((self._num_epochs, self._num_rollouts))
    performance = np.zeros((self._num_epochs, self._num_rollouts))
    for i in pb_epoch(range(self._num_epochs)):
      # Collect trajectories in environment.
      observations_het = []  # Heterogeneous data.
      actions = []
      returns = []
      logps_het = []  # Heterogeneous data.
      for j in pb_rollout(range(self._num_rollouts)):
        os, as_, rs, lps, exp, perf = self.rollout()
        observations_het.extend(os)
        actions.extend(as_)
        returns.extend(np.cumsum(rs[::-1])[::-1])
        logps_het.extend(lps)
        exploration[i, j] = exp
        performance[i, j] = perf

      # Pad graphs.
      sizes = np.array([
        (len(graph.nodes), len(graph.edges)) for graph in observations_het
      ]).max(0)
      sizes[0] = int(2 ** np.ceil(np.log2(sizes[0]))) + 1
      sizes[1] = int(2 ** np.ceil(np.log2(sizes[1])))
      num_edges = sizes[1]
      observations = [
        jraph.pad_with_graphs(graph, *sizes) for graph in observations_het
      ]

      # Pad log-probabilities.
      num_steps = len(logps_het)
      logps = np.ones((num_steps, num_edges)) * (-np.inf)
      mask = np.zeros((num_steps, num_edges), bool)
      for t in range(num_steps):
        logps[t, :len(logps_het[t])] = logps_het[t]
        mask[t, :len(logps_het[t])] = True

      # Shuffle and trim data.
      idx = self._rng.permutation(len(observations))[
        :(num_steps // self._batch_size) * self._batch_size
      ]
      observations = [observations[i] for i in idx]
      actions = jnp.asarray(actions)[idx]
      returns = jnp.asarray(returns)[idx]
      logps = jnp.asarray(logps)[idx]
      mask = jnp.asarray(mask)[idx]
      batches = [
        jraph.batch(
          observations[k * self._batch_size : (k + 1) * self._batch_size]
        ) for k in range(len(observations) // self._batch_size)
      ]

      # Update the actor.
      for j in pb_actor(range(self._num_actor_updates)):
        for k in range(len(observations) // self._batch_size):
          indices = slice(k * self._batch_size, (k + 1) * self._batch_size)
          batch = batches[k]
          params, opt_state, loss, kl = self.actor_update(
            self._actor_params,
            self._actor_opt_state,
            self._critic_params,
            batch,
            actions[indices],
            logps[indices],
            mask[indices],
            num_edges
          )
          if kl > self._kl_stop:
            (pb_actor.func.write if pba else print)(
              'KL early stopping after '
              f'{j * len(observations) // self._batch_size + k} updates.'
            )
            break  # Early stopping if KL divergence is too large.
          self._actor_params = params
          self._actor_opt_state = opt_state
          actor_losses_[i].append(loss)
        else:
          continue
        break

      # Update the critic.
      for j in pb_actor(range(self._num_critic_updates)):
        for k in range(len(observations) // self._batch_size):
          indices = slice(k * self._batch_size, (k + 1) * self._batch_size)
          batch = batches[k]
          params, opt_state, loss = self.critic_update(
            self._critic_params,
            self._critic_opt_state,
            batch,
            actions[indices],
            returns[indices],
            num_edges
          )
          self._critic_params = params
          self._critic_opt_state = opt_state
          critic_losses_[i].append(loss)

    actor_losses = np.full(
      (self._num_epochs, max(len(row) for row in actor_losses_)), np.nan
    )
    critic_losses = np.full(
      (self._num_epochs, max(len(row) for row in critic_losses_)), np.nan
    )
    for i in range(self._num_epochs):
      actor_losses[i, :len(actor_losses_[i])] = actor_losses_[i]
      critic_losses[i, :len(critic_losses_[i])] = critic_losses_[i]
    return actor_losses, critic_losses, exploration, performance

  @functools.partial(jax.jit, static_argnums=(0,))
  def q(self, params, features):
    return self._critic.apply(params, features).edges.ravel()

  @functools.partial(jax.jit, static_argnums=(0,))
  def policy(self, params, features):
    return (
      self._actor.apply(params, features).edges.ravel()
      * self._exploration_inv_temperature
    )

  def act(self, state):
    state, features, _, parcel, trucks, truck_ids = self.get_features(state)
    if parcel is None:
      return state, None, None
    if len(trucks) == 1:
      return state, parcel[0], trucks[truck_ids[0]]
    features = graph_utils.pad_graphs_tuple(features)
    logits = self.policy(self._actor_params, features)[jnp.array(truck_ids)]
    truck = trucks[truck_ids[np.argmax(logits)]]
    return state, parcel[0], truck


class GraphNet(nn.Module):
  """Encode-process-decode graph net architecture."""

  process_steps: int

  @nn.compact
  def __call__(self, state: jraph.GraphsTuple):
    def jraph_mlp(layers, residual=0):
      return jraph.concatenated_args(MLP(layers, residual))

    encoder = jraph.GraphNetwork(
        update_edge_fn=jraph_mlp([16, 16]),
        update_node_fn=jraph_mlp([16, 16]),
    )
    processor = jraph.GraphNetwork(
        update_edge_fn=jraph_mlp([16, 16], residual=True),
        update_node_fn=jraph_mlp([16, 16], residual=True),
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
  residual: bool

  @nn.compact
  def __call__(self, x: jax.Array):
    y = x[:, :self.layers[-1]] if self.residual else 0
    for features in self.layers[:-1]:
      x = nn.Dense(features)(x)
      x = nn.silu(x)
    x = nn.Dense(self.layers[-1])(x)
    return x + y


# if __name__ == '__main__':
#   rng = np.random.default_rng(42)
#   key = jax.random.PRNGKey(42)

#   # Initialize training MDP.
#   env = mdp.MiddleMileMDP(
#       num_hubs=10,
#       timesteps=50,
#       num_trucks_per_step=10,
#       max_truck_duration=5,
#       num_parcels=200,
#       mean_route_length=10,
#       cut_capacities=0,
#       unit_weights=True,
#       unit_capacities=True,
#   )

#   from tqdm import tqdm
#   controller = GNN_PPO(rng, key, env, 30, 1, 10, 10, 0.001, 0.1, 2, batch_size=256)
#   losses_actor, losses_critic, exploration, performance = controller.train(
#     pb_epoch=tqdm,
#     pb_rollout=functools.partial(tqdm, leave=False),
#     pb_actor=functools.partial(tqdm, leave=False),
#     pb_critic=functools.partial(tqdm, leave=False),
#   )

#   breakpoint()

#   state, _ = env.reset(rng)
#   _, features, *_ = controller.get_features(state)
#   controller.q(controller._critic_params, features)
#   controller.policy(controller._actor_params, features)

#   import matplotlib.pyplot as plt
#   plt.plot(performance.mean(1))
#   plt.show()  # it seems to learn?? increase the number of epochs further.
