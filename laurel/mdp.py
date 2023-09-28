"""Definition of the Middle-Mile MDP."""

from collections.abc import Mapping, Sequence

import numpy as np

from laurel import graph_utils


class MiddleMileMDP:
  """Markov decision process for the middle-mile logistics problem."""

  def __init__(
      self,
      num_hubs: int,
      timesteps: int,
      num_trucks_per_step: int,
      max_truck_duration: int,
      num_parcels: int | Sequence[int],
      mean_route_length: int,
      max_truck_capacity: float = 1,
      unit_capacities: bool = False,
      truck_sampling_inv_temp: float = 0.01,
      network_connectivity: int = 2,
      new_edge_p: float = 0.2,
      min_parcel_weight: float = 0.01,
      max_parcel_weight: float = 1,
      parcel_weight_shape: float = 0.1,
      unit_weights: bool = False,
      start_inv_temp: float = 0.1,
      dist_inv_temp: float = 0.1,
      max_tries: int = 50,
      cut_capacities: float = 0,
      static_network: bool = False,
  ):
    """Initializes middle-mile MDP.

    Args:
      num_hubs: Number of depots in the network.
      timesteps: Number of time steps.
      num_trucks_per_step: Number of trucks leaving every time step
        (deterministic).
      max_truck_duration: The maximal number of time steps a truck takes to
        reach its destination. The actual number is sampled uniformly from {1,
        ..., `max_truck_duration`}.
      num_parcels: Number of parcels to include in the network.
      mean_route_length: Average length of a parcel route (in time steps). After
        each time step of a parcel route, it is randomly decided whether to
        terminate the route, with probability 1 / `mean_route_length`. This way,
        a parcel route lasts `sum((1 - 1 / mean_route_length)**k)` time steps in
        expectation. This geometric sum approaches `mean_route_length` as the
        number of time steps in `state` becomes larger.
      max_truck_capacity: The maximum scalar capacity of trucks. The actual
        capacity is sampled uniformly from [0, `max_truck_capacity`).
      unit_capacities: If `True`, all truck capacities are set to 1. The
        argument `max_truck_capacity` is ignored in this case.
      truck_sampling_inv_temp: When sampling the trucks for a given time step,
        probabilities for the trucks are given by
        `softmax(truck_sampling_inv_temp * (degree_truck_sender +
        degree_truck_receiver))`. This way, trucks connecting two high-degree
        nodes are sampled more frequently.
      network_connectivity: The `m` parameter of the extended Barabási-Albert
        model [1].
      new_edge_p: The `p` parameter of the extended Barabási-Albert model [1].
      min_parcel_weight: The scale (`m`) parameter of the Pareto distribution
        [2] from which parcel weights are sampled.
      max_parcel_weight: The maximum weight of parcels. The parcel weights are
        sampled from a truncated Pareto distribution using rejection sampling to
        keep weights below `max_parcel_weight`.
      parcel_weight_shape: The shape (`a`) parameter of the Pareto distribution
        [2] from which parcel weights are sampled.
      unit_weights: If `True`, all parcel weights are set to 1 instead of being
        sampled from a Pareto distribution.
      start_inv_temp: The start node hub is sampled from a Boltzmann
        distribution with exponent `-start_inv_temp * degree(hub)` to preferably
        start at hubs with lower degree.
      dist_inv_temp: When sampling parcel routes it is often posssible to choose
        between multiple trucks. In these cases, the truck is chosen among the
        available trucks by sampling from a Boltzmann distribution with exponent
        `dist_inv_temp * dist_from_start` (where the distance to the start node
        is measured in terms of resistance distance, i.e. `distances`). This
        encourages parcels to go further from their start nodes.
      max_tries: If, after a parcel route has been sampled, it is found that
        this parcel route doesn't actually use any real trucks (instead, the
        parcel just stays at its start hub), we try again. This is done for a
        maximum of `max_tries` times (after which the 'staying' parcel route
        will simply be accepted). The parcel weight is reduced by 10% each try
        to make it easier to place the parcel.
      cut_capacities: After all parcels have been placed in the network, there
        is normally still capacity left in some trucks. `cut_capacities` is a
        number between 0 and 1 that specifies how much of this excess capacity
        should be removed (by reducing the trucks' capacities). Thus, 1 means
        all excess capacity is cut and 0 means no capacity is cut.
      static_network: If `True`, the logistics network will be created in the
        first call of `reset`. Subsequent calls of `reset` will only sample new
        parcels, but use the same logistics network.
    """
    self._num_hubs = num_hubs
    self._timesteps = timesteps
    self._num_trucks_per_step = num_trucks_per_step
    self._max_truck_duration = max_truck_duration
    self._num_parcels = num_parcels
    self._mean_route_length = mean_route_length
    self._max_truck_capacity = max_truck_capacity
    self._unit_capacities = unit_capacities
    self._truck_sampling_inv_temp = truck_sampling_inv_temp
    self._network_connectivity = network_connectivity
    self._new_edge_p = new_edge_p
    self._min_parcel_weight = min_parcel_weight
    self._max_parcel_weight = max_parcel_weight
    self._parcel_weight_shape = parcel_weight_shape
    self._unit_weights = unit_weights
    self._start_inv_temp = start_inv_temp
    self._dist_inv_temp = dist_inv_temp
    self._max_tries = max_tries
    self._cut_capacities = cut_capacities
    self._static_network = static_network
    self._empty_state = None
    self._network = None
    self._distances = None
    self._base_connections = None
    self._state = None
    self._solution = None
    self._edge_parcels = None
    self._connections = None

  def reset(
      self, rng: np.random.Generator
  ) -> tuple[graph_utils.Graph, graph_utils.SolutionDict]:
    """Resets the MDP state.

    Must be called at the start of each episode to initialize the MDP state.

    Args:
      rng: Random number generator.

    Returns:
      The new MDP state.
    """
    num_parcels = (
      self._num_parcels
      if np.isscalar(self._num_parcels)
      else rng.choice(self._num_parcels)
    )
    if self._network is None or not self._static_network:
      # Sample logistics network and prune skippable nodes/edges.
      self._empty_state, self._network, self._distances = (
          graph_utils.make_random_network(
              rng=rng,
              num_hubs=self._num_hubs,
              timesteps=self._timesteps,
              num_trucks_per_step=self._num_trucks_per_step,
              max_truck_duration=self._max_truck_duration,
              max_truck_capacity=self._max_truck_capacity,
              unit_capacities=self._unit_capacities,
              truck_sampling_inv_temp=self._truck_sampling_inv_temp,
              network_connectivity=self._network_connectivity,
              new_edge_p=self._new_edge_p,
          )
      )
      self._empty_state, self._base_connections, *_ = graph_utils.prune_network(
          self._empty_state, prune_parcels=False
      )

    # Sample parcels and prune unreachable parts of graph.
    self._state, self._solution = graph_utils.make_random_parcels(
        rng=rng,
        state=self._empty_state,
        network=self._network,
        distances=self._distances,
        num_parcels=num_parcels,
        mean_route_length=self._mean_route_length,
        min_parcel_weight=self._min_parcel_weight,
        max_parcel_weight=self._max_parcel_weight,
        parcel_weight_shape=self._parcel_weight_shape,
        unit_weights=self._unit_weights,
        start_inv_temp=self._start_inv_temp,
        dist_inv_temp=self._dist_inv_temp,
        max_tries=self._max_tries,
        cut_capacities=self._cut_capacities,
    )
    self._state, connections, self._edge_parcels, self._solution = (
        graph_utils.prune_network(self._state, solution=self._solution)
    )
    self._connections = dict(connections) | dict(self._base_connections)
    assert self._solution is not None  # Needed for Pytype.
    return self._state, self._solution

  def get_next_parcel(
      self,
      state: graph_utils.Graph | None = None,
      prune_parcel: bool = True,
      last_parcel: bool = False,
  ) -> tuple[graph_utils.Graph, int | None, Sequence[int] | None]:
    """Gets the next parcel to route.

    Retrieves the earliest (default) or last parcel in the network which has
    available trucks.

    Args:
      state: Time-expanded logistics network. If not specified, the internal
        state of the MDP instance is used, for which `reset` has to have been
        called.
      prune_parcel: Whether, once a parcel has been found, to only return those
        trucks which are part of valid paths to the parcel's goal.
      last_parcel: Whether to retrieve the last parcel in the network instead of
        the earliest one.

    Returns:
      A triple `(state, parcel, trucks)`, where:
      - `state` is the MDP state which might be different than the one that is
        passed. When parcels are encountered which have no available trucks,
        they are removed using `clean_step`. The cleaned state is returned.
      - `parcel` is the index of the parcel to route in `state`. If no suitable
        parcel was found, this is `None`.
      - `trucks` are the trucks immediately available for `parcel`, as edge
        indices in `state`. If there is no suitable parcel, this is `None`.

    Raises:
      MDPInitializationError: If the MDP state has not been initialized, i.e. if
        `reset` has not been called.
    """
    state, update_state = self._check_state(state)

    while True:
      # Retrieve all parcels in state.
      parcels = state.edges[:, graph_utils.EdgeFeatures.PARCEL_FORWARD] == 1

      if not parcels.any():
        # No parcels found.
        return state, None, None

      # Get the earliest or the last parcel.
      operator = np.argmax if last_parcel else np.argmin
      parcel = np.where(parcels)[0][
          operator(state.nodes[state.senders[parcels]][:, 1])
      ].item()

      # Check if this parcel is valid. Otherwise, try again.
      state, parcel, trucks = self.get_actions(
          state, parcel, prune_parcel, last_parcel
      )
      if update_state:
        self._state = state
      if trucks is not None:
        return state, parcel, trucks

  def get_actions(
      self,
      state: graph_utils.Graph | None = None,
      parcel: int | None = None,
      prune_parcel: bool = True,
      last_parcel: bool = False,
  ) -> tuple[graph_utils.Graph, int | None, Sequence[int] | None]:
    """Retrieves all available trucks for `parcel`.

    Args:
      state: Time-expanded logistics network. If not specified, the internal
        state of the MDP instance is used, for which `reset` has to have been
        called.
      parcel: The index of the forward-in-time parcel edge in `state` for which
        to search for trucks. If not specified, the next available parcel is
        retrieved using `get_next_parcel`.
      prune_parcel: Whether to only return those trucks which are part of valid
        paths to the parcel's goal.
      last_parcel: Whether to retrieve the last parcel in the network instead of
        the earliest one (in case `parcel` is `None`).

    Returns:
      A triple `(state, parcel, trucks)`, where:
      - `state` is the MDP state which might be different than the one that is
        passed. When parcels are encountered which have no available trucks,
        they are removed using `clean_step`. The cleaned state is returned.
      - `parcel` is the index of the parcel to route in `state`. If no suitable
        parcel was found, this is `None`.
      - `trucks` are the trucks immediately available for `parcel`, as edge
        indices in `state`. If there is no suitable parcel, this is `None`.

    Raises:
      MDPInitializationError: If the MDP state has not been initialized, i.e. if
        `reset` has not been called.
    """
    if parcel is None:
      return self.get_next_parcel(state, prune_parcel, last_parcel)

    state, update_state = self._check_state(state)

    # Get all available trucks for parcel.
    if prune_parcel:
      # Only consider tucks which lead to the goal.
      _, edges = graph_utils.prune_parcel(state, parcel)
    else:
      edges = np.arange(len(state.edges))
    trucks = (state.senders[edges] == state.senders[parcel]) & (
        (
            state.edges[edges, graph_utils.EdgeFeatures.TRUCK_FORWARD] == 1
        )  # Real trucks.
        & (
            state.edges[edges, graph_utils.EdgeFeatures.TRUCK_CAPACITY]
            >= state.edges[parcel, graph_utils.EdgeFeatures.PARCEL_WEIGHT]
        )  # Sufficient capacity (in real trucks).
        | (
            state.edges[edges, graph_utils.EdgeFeatures.VIRTUAL_FORWARD] == 1
        )  # Virtual trucks.
    )

    # If there are trucks available, return them.
    if trucks.any():
      return state, parcel, edges[trucks].tolist()

    # Otherwise, remove this parcel from the state.
    state, parcel = graph_utils.clean_step(
        state, parcel, state.senders[parcel], True
    )
    if update_state:
      self._state = state
    return state, parcel, None

  def get_feature_graph(
      self,
      num_steps: int,
      state: graph_utils.Graph | None = None,
      parcel: int | None = None,
      only_reachable: bool = False,
      min_phantom_weight: float | None = 0,
      prune_parcel: bool = True,
      last_parcel: bool = False,
  ) -> tuple[
      graph_utils.Graph,
      graph_utils.Graph | None,
      tuple[int, int] | None,
      Mapping[int, int] | None,
  ]:
    """Extracts feature graph for next parcel to consider.

    First, starts at parcel start and goal nodes and constructs a base graph
    containing all nodes and edges reachable in `num_steps` steps. Then, adds
    virtual connections to this graph where possible to connect the top and
    bottom parts. Finally, node and edge features are added to the feature
    graph:
      - Node features:
        - Resistance distance: The resistance distance of a node's hub to the
          goal node's hub in the non-expanded network. The edge-conductances are
          given by `truck_sampling_inv_temp * (degree_truck_sender +
          degree_truck_receiver)` to relate to the frequency of trucks along the
          edges (see `graph_utils.make_random_network`). This feature is the
          third node feature (after the hub index and the time index).
        - Relative time: Where a node lies in time between the start node
          (relative time = 0) and the end node (relative time = 1). Given by
          `(time - start_time) / (end_time - start_time)` if `end_time >
          start_time` and zero otherwise. This feature is the fourth node
          feature.
      - Edge features:
        - Phantom parcel weight: The feature graph normally only contains a
          small subset of all parcels. The removed parcels might still be
          relevant to the decision at hand, because they may share a truck that
          is included in the feature graph. This edge feature is the sum of all
          removed parcels' weights, weighted by how likely they are to use the
          truck, given that they follow a uniformly random policy. The features
          use the `PARCEL_WEIGHT` index and are only computed for real trucks.

    Args:
      num_steps: Number of expansion steps (starting from the parcel start and
        goal nodes) to construct the feature graph.
      state: The state for which to search for actions. If not specified, the
        internal state of the MDP instance is used, for which `reset` has to
        have been called.
      parcel: Parcel (as the forward-in-time edge index in `state`) for which to
        build the feature graph. If not specified, the earliest parcel in the
        graph will be retrieved using `get_next_parcel`.
      only_reachable: Whether to remove all nodes and edges from the feature
        graph which are not direcly relevant for the parcel under consideration
        (i.e. those that would be removed by `graph_utils.prune_parcel`).
      min_phantom_weight: When phantom weights reach this minimum weight, they
        are not further split along future trucks. This can improve performance
        at the cost of omitting some information. If `None`, phantom weights are
        not computed at all.
      prune_parcel: Whether to only return those trucks which are part of valid
        paths to the parcel's goal.
      last_parcel: Whether to retrieve the last parcel in the network instead of
        the earliest one (in case `parcel` is `None`).

    Returns:
      - The MDP state, which might be different than the one that is passed.
        When parcels are encountered which have no available trucks, they are
        removed using `clean_step`. The cleaned state is returned.
      - The feature graph (`graph_utils.Graph` object), or `None` if no parcels
        are found in the network.
      - Parcel tuple: (parcel index in `state`, parcel index in feature graph),
        or `None` if no parcels are found in the network.
      - A dictionary mapping indices of actions (available trucks for parcel)
        from their edge-indices in the feature graph to their edge-indices in
        `state`, or `None` if no parcels are found in the network.

    Raises:
      MDPInitializationError: If the MDP state has not been initialized, i.e. if
        `reset` has not been called.
    """
    state, update_state = self._check_state(state)
    # Assured by self._check_state, but this assertion is necessary for pytype.
    assert self._distances is not None

    # Get parcel and available actions.
    state, parcel, action_trucks = self.get_actions(
        state, parcel, prune_parcel, last_parcel
    )
    if update_state:
      self._state = state
    if parcel is None:
      return state, None, None, None

    start = state.senders[parcel]
    goal = state.receivers[parcel]

    # Construct base graph by moving along edges, starting at the parcel start
    # and goal nodes.
    reachable_edges = None
    if only_reachable:
      _, reachable_edges = graph_utils.prune_parcel(state, parcel)
    nodes = {start, goal}
    edges = {parcel}
    parcels = {parcel}
    open_nodes = {start, goal}
    for _ in range(num_steps):
      next_open = set()
      for node in open_nodes:
        # Expand node: find all trucks and parcels ending or starting here.
        truck_edges = np.where(
            ((state.senders == node) | (state.receivers == node))
            & (
                (
                    state.edges[:, graph_utils.EdgeFeatures.TRUCK_FORWARD] == 1
                )  # Real trucks.
                | (
                    state.edges[:, graph_utils.EdgeFeatures.VIRTUAL_FORWARD]
                    == 1
                )  # Virtual forward truck.
            )
        )[0]
        parcel_edges = np.where(
            ((state.senders == node) | (state.receivers == node))
            & (state.edges[:, graph_utils.EdgeFeatures.PARCEL_FORWARD] == 1)
        )[0]
        if only_reachable:
          truck_edges = truck_edges[np.isin(truck_edges, reachable_edges)]

        # Add parcels to graph.
        for parcel_edge in parcel_edges:
          parcels.add(
              int(state.edges[parcel_edge, graph_utils.EdgeFeatures.ID])
          )
          edges.add(parcel_edge)
          nodes |= {state.senders[parcel_edge], state.receivers[parcel_edge]}

        # Add trucks to graph and find neighboring nodes to expand next.
        for truck_edge in truck_edges:
          if state.senders[truck_edge] not in nodes:
            next_open.add(state.senders[truck_edge])
          if state.receivers[truck_edge] not in nodes:
            next_open.add(state.receivers[truck_edge])
          edges.add(truck_edge)
          nodes |= {state.senders[truck_edge], state.receivers[truck_edge]}
      open_nodes = next_open

    edges = np.array(list(edges))
    nodes = np.array(list(nodes))

    # Add reverse truck and parcel edges.
    edges = np.where(
        state.edges[:, graph_utils.EdgeFeatures.ID]
        == state.edges[edges, graph_utils.EdgeFeatures.ID][:, None]
    )[1]

    # Add virtual connections to graph where possible to connect start and goal
    # parts.
    new_edges = []
    new_receivers = []
    new_senders = []
    locations = np.unique(state.nodes[nodes, 0])
    first_id = state.edges[:, graph_utils.EdgeFeatures.ID].max() + 1
    for location in locations:
      # Collect all nodes in feature graph belonging to this hub.
      loc_nodes = np.where(state.nodes[nodes, 0] == location)[0]
      if len(loc_nodes) == 1:
        continue  # No need for virtual connections if only one node exists.
      times = state.nodes[nodes[loc_nodes], 1]
      idx = np.argsort(times)  # Sort nodes by time.

      for i in range(len(loc_nodes) - 1):
        # Check if there is already a virtual connection.
        virtual_truck = np.where(
            (state.senders[edges] == nodes[loc_nodes][idx[i]])
            & (state.receivers[edges] == nodes[loc_nodes][idx[i + 1]])
            & (
                state.edges[edges, graph_utils.EdgeFeatures.VIRTUAL_FORWARD]
                == 1
            )
        )[0]
        if virtual_truck.size > 0:
          continue  # If yes, on to the next node.

        # Add virtual connection.
        truck = np.zeros(len(graph_utils.EdgeFeatures))
        reverse_truck = np.zeros(len(graph_utils.EdgeFeatures))
        truck[graph_utils.EdgeFeatures.ID] = first_id + len(new_edges) // 2
        truck[graph_utils.EdgeFeatures.VIRTUAL_FORWARD] = 1
        reverse_truck[graph_utils.EdgeFeatures.ID] = (
            first_id + len(new_edges) // 2
        )
        reverse_truck[graph_utils.EdgeFeatures.VIRTUAL_BACKWARD] = 1
        new_edges += [truck, reverse_truck]
        new_senders += [nodes[loc_nodes][idx[i]], nodes[loc_nodes][idx[i + 1]]]
        new_receivers += [
            nodes[loc_nodes][idx[i + 1]],
            nodes[loc_nodes][idx[i]],
        ]

    # Add resistance distance and relative time node features.
    distances = self._distances[state.nodes[nodes, 0], state.nodes[goal, 0]][
        :, None
    ]
    if state.nodes[start, 1] < state.nodes[goal, 1]:
      times = (
          (state.nodes[nodes, 1] - state.nodes[start, 1])
          / (state.nodes[goal, 1] - state.nodes[start, 1])
      )[:, None]
    else:
      # If start node time >= goal node time, relative time doesn't make sense.
      # Just make the node feature 0 for all nodes.
      times = np.zeros_like(distances)

    # Add "phantom parcel weight" edge features.
    # Get all ids of real truck edges in the feature graph.
    trucks = set(
        state.edges[
            edges[
                state.edges[edges, graph_utils.EdgeFeatures.TRUCK_FORWARD] == 1
            ],
            graph_utils.EdgeFeatures.ID,
        ].astype(int)
    )

    # Find parcels relevant to graph, but remove those included in feature graph
    # or already delivered. If `min_phantom_weight` is None, we don't add
    # phantom parcels to the feature graph.
    parcel_ids = set()
    if min_phantom_weight is not None:
      for truck in trucks:
        possible_parcels = self._edge_parcels[truck]
        parcel_ids |= {
            parcel_id
            for parcel_id in possible_parcels
            if parcel_id in state.edges[:, graph_utils.EdgeFeatures.ID]
        }
      parcel_ids -= parcels

    # Compute phantom parcel weights.
    weights = np.zeros(len(edges))
    for parcel_id in parcel_ids:
      # Get pruned parcel-specific graph.
      parcel_edge = np.where(
          (state.edges[:, graph_utils.EdgeFeatures.ID] == parcel_id)
          & (state.edges[:, graph_utils.EdgeFeatures.PARCEL_FORWARD] == 1)
      )[0][0]
      _, parcel_edges = graph_utils.prune_parcel(state, parcel_edge)

      # Compute phantom weights on this graph by walking along the pruned
      # parcel-specific graph and distributing the parcel weight over the
      # encountered edges.
      # Start from the parcel start node with the full parcel weight.
      open_nodes = {
          state.senders[parcel_edge]: state.edges[
              parcel_edge, graph_utils.EdgeFeatures.PARCEL_WEIGHT
          ]
      }
      while open_nodes:
        node, weight = open_nodes.popitem()
        if weight < min_phantom_weight:
          continue  # Break parcel off here, expected weight is too small.

        # Find all trucks from this node.
        trucks_from_node = parcel_edges[
            (state.senders[parcel_edges] == node)
            & (
                (
                    state.edges[
                        parcel_edges, graph_utils.EdgeFeatures.TRUCK_FORWARD
                    ]
                    == 1
                )
                | (
                    state.edges[
                        parcel_edges, graph_utils.EdgeFeatures.VIRTUAL_FORWARD
                    ]
                    == 1
                )
            )
        ]
        if trucks_from_node.size == 0:
          # If there are no trucks, we are done.
          continue

        # Divide weight equally among options (uniformly random policy).
        weight = weight / len(trucks_from_node)

        for truck in trucks_from_node:
          # Add weight to truck and to reverse truck (if not virtual).
          if state.edges[truck, graph_utils.EdgeFeatures.TRUCK_FORWARD] == 1:
            weights[
                state.edges[edges, graph_utils.EdgeFeatures.ID]
                == state.edges[truck, graph_utils.EdgeFeatures.ID]
            ] += weight

          # Add receiving node to be processed.
          if state.receivers[truck] in open_nodes:
            open_nodes[state.receivers[truck]] += weight
          else:
            open_nodes[state.receivers[truck]] = weight

    # Add phantom parcels and new virtual connections to feature graph edges.
    edge_features = state.edges[edges]
    edge_features[:, graph_utils.EdgeFeatures.PARCEL_WEIGHT] += weights
    if new_edges:
      edge_features = np.concatenate([edge_features, new_edges])

    # Translate parcel and action trucks into edge indices of feature graph.
    actions = {
        np.where(edges == truck)[0][0]: truck
        for truck in action_trucks
        if truck in edges
    }
    parcel = (parcel, np.where(edges == parcel)[0][0])

    # Build Graph representation.
    receivers = np.concatenate([state.receivers[edges], new_receivers])
    senders = np.concatenate([state.senders[edges], new_senders])
    receivers = np.where(nodes == receivers[:, None])[1]
    senders = np.where(nodes == senders[:, None])[1]
    return (
        state,
        graph_utils.Graph(
            np.hstack([state.nodes[nodes], distances, times]),
            edge_features,
            receivers,
            senders,
        ),
        parcel,
        actions,
    )

  def step(
      self,
      parcel: int,
      truck: int,
      state: graph_utils.Graph | None = None,
      prune: bool = True,
  ) -> tuple[graph_utils.Graph, bool, int | None]:
    """Does one step of the environment dynamics: puts parcel into truck.

    Args:
      parcel: The parcel to move, as the index of the parcel-edge going forward
        in time.
      truck: The truck to which the parcel should be allocated, as the index of
        the truck/virtual-edge going forward in time.
      state: The state to which the decision is applied. If not specified, the
        internal state of the MDP instance is used, for which `reset` has to
        have been called. If the internal state is used, it is updated in this
        function. If `state` is specified, the internal state is not updated
        here.
      prune: Whether to remove nodes and edges from the graph which become
        superfluous once the parcel has been moved.

    Returns:
      The new MDP state together with a bool indicating whether a delivery has
      been made in this step and the new index `parcel` in the changed state. If
      the parcel has been delivered this new index is `None`.

    Raises:
      MDPInitializationError: If the MDP state has not been initialized, i.e. if
        `reset` has not been called.
    """
    state, update_state = self._check_state(state)

    # Variables for building next state.
    edges = state.edges.copy()
    receivers = state.receivers.copy()
    senders = state.senders.copy()

    # Get reverse edges for given truck and parcel.
    reverse_truck = np.where(
        (
            (edges[:, graph_utils.EdgeFeatures.TRUCK_BACKWARD] == 1)
            | (edges[:, graph_utils.EdgeFeatures.VIRTUAL_BACKWARD] == 1)
        )
        & (
            edges[:, graph_utils.EdgeFeatures.ID]
            == edges[truck, graph_utils.EdgeFeatures.ID]
        )
    )[0][0]
    reverse_parcel = np.where(
        (edges[:, graph_utils.EdgeFeatures.PARCEL_BACKWARD] == 1)
        & (
            edges[:, graph_utils.EdgeFeatures.ID]
            == edges[parcel, graph_utils.EdgeFeatures.ID]
        )
    )[0][0]

    # Change parcel location.
    senders[parcel] = receivers[truck]
    receivers[reverse_parcel] = receivers[truck]

    # Change truck capacity (not for virtual trucks).
    if edges[truck, graph_utils.EdgeFeatures.TRUCK_FORWARD]:
      edges[truck, graph_utils.EdgeFeatures.TRUCK_CAPACITY] -= edges[
          parcel, graph_utils.EdgeFeatures.PARCEL_WEIGHT
      ]
      edges[reverse_truck, graph_utils.EdgeFeatures.TRUCK_CAPACITY] -= edges[
          parcel, graph_utils.EdgeFeatures.PARCEL_WEIGHT
      ]

    # Check if parcel has reached destination.
    delivery = senders[parcel] == receivers[parcel]

    # Remove delivered parcel from graph.
    state = graph_utils.Graph(state.nodes, edges, receivers, senders)
    remove_parcel = (
        delivery
        or state.nodes[senders[parcel], 1] > state.nodes[receivers[parcel], 1]
    )
    state, parcel = graph_utils.clean_step(
        state, parcel, state.senders[truck].item(), remove_parcel, prune
    )

    if update_state:
      self._state = state
    return state, delivery, parcel

  def _check_state(
      self, state: graph_utils.Graph | None
  ) -> tuple[graph_utils.Graph, bool]:
    """Checks if state has been initialized.

    Args:
      state: State to check. If `None`, the internal MDP state is used.

    Returns:
      The original state if it is not `None`, otherwise the internal MDP state,
      and a bool specifying whether the internal state is used.

    Raises:
      MDPInitializationError: If the internal MDP state has not been
        initialized, i.e. if `reset` has not been called.
    """
    internal_state = state is None
    state = state if state is not None else self._state
    if self._state is None:
      raise MDPInitializationError(
          "The MDP state has not been initialized. Please call reset to do so."
      )
    return state, internal_state


class MDPInitializationError(Exception):
  """Error raised when the MDP state has not been initialized using `reset`."""
