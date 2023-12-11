"""Functions for working with graphs."""

from __future__ import annotations

import collections
from collections.abc import Mapping, Sequence
import dataclasses
import enum
from typing import Any, Optional

import jax.numpy as jnp
import jraph
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy import special


SolutionDict = Mapping[int, Sequence[tuple[int, int]]]


@enum.unique
class EdgeFeatures(enum.IntEnum):
  """Edge feature indices."""

  ID = 0
  TRUCK_FORWARD = 1
  TRUCK_BACKWARD = 2
  PARCEL_FORWARD = 3
  PARCEL_BACKWARD = 4
  VIRTUAL_FORWARD = 5
  VIRTUAL_BACKWARD = 6
  TRUCK_CAPACITY = 7
  PARCEL_WEIGHT = 8


@dataclasses.dataclass
class Graph:
  """NumPy alternative for single-graph instances of jraph.GraphsTuple."""

  nodes: np.ndarray
  edges: np.ndarray
  receivers: np.ndarray
  senders: np.ndarray

  @classmethod
  def from_jraph(cls, network: jraph.GraphsTuple) -> Graph:
    if len(network.n_node) > 1:
      raise ValueError(
          'The `GraphsTuple` object contains multiple graphs. This class only'
          ' supports single-graph instances.'
      )
    return cls(
        nodes=np.asarray(network.nodes),
        edges=np.asarray(network.edges),
        receivers=np.asarray(network.receivers),
        senders=np.asarray(network.senders),
    )

  def to_jraph(self, padded: bool = False) -> jraph.GraphsTuple:
    """Convert to jraph.GraphsTuple.

    Args:
      padded: Whether to pad the number of nodes and edges of the graphs tuple
        to the respective nearest powers of two.
    
    Returns:
      The jraph.GraphsTuple representation of this graph.
    """
    graphs_tuple = jraph.GraphsTuple(
        nodes=jnp.asarray(self.nodes),
        edges=jnp.asarray(self.edges),
        receivers=jnp.asarray(self.receivers),
        senders=jnp.asarray(self.senders),
        globals=None,
        n_node=jnp.array([len(self.nodes)]),
        n_edge=jnp.array([len(self.edges)]),
    )
    if padded:
      graphs_tuple = pad_graphs_tuple(graphs_tuple)
    return graphs_tuple


def pad_graphs_tuple(graphs_tuple: jraph.GraphsTuple) -> jraph.GraphsTuple:
  """Pads jraph.GraphsTuple nodes and edges to nearest power of two.

  For example, if a `GraphsTuple` has 7 nodes, 5 edges and 3 graphs, this method
  would pad the `GraphsTuple` nodes and edges:
    7 nodes --> 8 nodes (2^3)
    5 edges --> 8 edges (2^3)
  And since padding is accomplished using `jraph.pad_with_graphs`, an extra
  graph and node is added:
    8 nodes --> 9 nodes
    3 graphs --> 4 graphs

  Args:
    graphs_tuple: a batched `GraphsTuple` (can be batch size 1).

  Returns:
    A graphs_tuple batched to the nearest power of two.

  Notes:
    Adapted from jraph/ogb_examples/train_flax.py.
  """
  # Add 1 since we need at least one padding node for pad_with_graphs.
  pad_nodes_to = int(2 ** np.ceil(np.log2(graphs_tuple.n_node.sum()))) + 1
  pad_edges_to = int(2 ** np.ceil(np.log2(graphs_tuple.n_edge.sum())))

  # Add 1 since we need at least one padding graph for pad_with_graphs.
  # We do not pad to nearest power of two because the batch size is fixed.
  pad_graphs_to = graphs_tuple.n_node.shape[0] + 1

  return jraph.pad_with_graphs(
      graphs_tuple, pad_nodes_to, pad_edges_to, pad_graphs_to
  )


def make_random_network(
    rng: np.random.Generator,
    num_hubs: int,
    timesteps: int,
    num_trucks_per_step: int,
    max_truck_duration: int,
    max_truck_capacity: float = 1,
    unit_capacities: bool = False,
    truck_sampling_inv_temp: float = 0.01,
    network_connectivity: int = 2,
    new_edge_p: float = 0.2,
) -> tuple[Graph, nx.Graph, np.ndarray]:
  """Generates a random time-expanded transportation network.

  First, generates a network of depots and connections between them according to
  the extended Barabási-Albert algorithm [1]. This results in a few highly
  connected (large degree), while most depots only have a few connections. The
  time-expanded representation is then generated from the original network by
  sampling a pre-specified number of trucks at each time step, each of which
  starts at the current time step and ends a random number of time steps later
  (taking care to not exceed the specified total number of time steps). Parcels
  may be added with the `make_random_parcels` function.

  Args:
    rng: Random number generator.
    num_hubs: Number of depots in the network.
    timesteps: Number of time steps.
    num_trucks_per_step: Number of trucks leaving every time step
      (deterministic).
    max_truck_duration: The maximal number of time steps a truck takes to reach
      its destination. The actual number is sampled uniformly from {1, ...,
      `max_truck_duration`}.
    max_truck_capacity: The maximum scalar capacity of trucks. The actual
      capacity is sampled uniformly from [0, `max_truck_capacity`).
    unit_capacities: If `True`, all truck capacities are set to 1. The argument
      `max_truck_capacity` is ignored in this case.
    truck_sampling_inv_temp: When sampling the trucks for a given time step,
      probabilities for the trucks are given by `softmax(truck_sampling_inv_temp
      * (degree_truck_sender + degree_truck_receiver))`. This way, trucks
      connecting two high-degree nodes are sampled more frequently.
    network_connectivity: The `m` parameter of the extended Barabási-Albert
      model [1].
    new_edge_p: The `p` parameter of the extended Barabási-Albert model [1].

  Returns:
    A triple `(graph, network, distances)`, where:
    - `graph` is the time-expanded network represented as a `Graph` object. The
      node features are tuples of the form (location, time). The edge features
      are of the form (edge id, forward truck?, backward truck?, forward
      parcel?, backward parcel?, forward virtual?, backward virtual?, capacity
      if truck, weight if parcel).
    - `network` is the non-expanded network as a NetworkX graph. This network
      has edge weights of `truck_sampling_inv_temp * (degree_sender +
      degree_receiver)`. As this is the probability of sampling a truck, the
      edge weights give information about the frequency of a connection.
    - `distances` is the symmetric `num_hubs` x `num_hubs` matrix of resistance
      distances between nodes in the non-expanded network. The resistances are
      all set to 1.

  References:
    [1]: NetworkX reference page for the extended Barabási-Albert algorithm:
      https://networkx.org/documentation/stable/reference/generated/networkx.generators.random_graphs.extended_barabasi_albert_graph.html.
  """
  # Generate transportation network.
  network = nx.extended_barabasi_albert_graph(
      num_hubs, network_connectivity, new_edge_p, 0, rng
  )

  # Compute resistance distances (only works on undirected network).
  distances = np.zeros((num_hubs, num_hubs))
  for i in range(num_hubs):
    for j in range(i + 1, num_hubs):
      distances[i, j] = distances[j, i] = nx.resistance_distance(network, i, j)

  # We work with directed networks.
  network = network.to_directed()
  edges = np.asarray(network.edges)

  # Truck frequencies depend on the importance of an edge (degree of sender +
  # degree of receiver).
  weights = [
      truck_sampling_inv_temp * (network.degree[a] + network.degree[b])
      for (a, b) in network.edges
  ]
  nx.set_edge_attributes(
      network,
      name='weight',
      values=dict(zip(network.edges, weights)),
  )

  # Node features for time-expanded network: (location, time).
  nodes = np.hstack([
      np.tile(np.arange(num_hubs), timesteps + 1)[:, None],
      np.repeat(np.arange(timesteps + 1), num_hubs)[:, None],
  ])

  # Trucks for time-expanded network (sample one time step at a time).
  truck_senders = []
  truck_receivers = []
  num_trucks = 0
  for timestep in range(timesteps):
    # Sample trucks and add to network.
    trucks = rng.choice(
        edges,
        (num_trucks_per_step,),
        replace=False,
        p=special.softmax(weights),
    )
    times = 1 + rng.choice(max_truck_duration, (num_trucks_per_step,))
    trucks = trucks[times <= timesteps - timestep]
    times = times[times <= timesteps - timestep]
    for truck, time in zip(trucks, times):
      truck_senders += [
          truck[0] + num_hubs * timestep,
          truck[1] + num_hubs * (timestep + time),
      ]
      truck_receivers += [
          truck[1] + num_hubs * (timestep + time),
          truck[0] + num_hubs * timestep,
      ]
    num_trucks += len(trucks)

  # Sample truck capacities and create edges.
  capacities = (
      np.ones(num_trucks)
      if unit_capacities
      else max_truck_capacity * rng.uniform(size=(num_trucks,))
  )
  truck_edges = np.zeros((2 * num_trucks, len(EdgeFeatures)))
  truck_edges[::2, EdgeFeatures.ID] = np.arange(num_trucks)
  truck_edges[1::2, EdgeFeatures.ID] = np.arange(num_trucks)
  truck_edges[::2, EdgeFeatures.TRUCK_FORWARD] = 1
  truck_edges[1::2, EdgeFeatures.TRUCK_BACKWARD] = 1
  truck_edges[::2, EdgeFeatures.TRUCK_CAPACITY] = capacities
  truck_edges[1::2, EdgeFeatures.TRUCK_CAPACITY] = capacities

  # Add virtual trucks ("stay at location") to network.
  virtual_edges = np.zeros((2 * timesteps * num_hubs, len(EdgeFeatures)))
  virtual_edges[::2, EdgeFeatures.ID] = num_trucks + np.arange(
      timesteps * num_hubs
  )
  virtual_edges[1::2, EdgeFeatures.ID] = num_trucks + np.arange(
      timesteps * num_hubs
  )
  virtual_edges[::2, EdgeFeatures.VIRTUAL_FORWARD] = 1
  virtual_edges[1::2, EdgeFeatures.VIRTUAL_BACKWARD] = 1
  virtual_senders = np.repeat(np.arange(num_hubs * timesteps), 2)
  virtual_senders[1::2] += num_hubs
  virtual_receivers = np.repeat(np.arange(num_hubs * timesteps), 2)
  virtual_receivers[::2] += num_hubs

  # Build Graph representation of network.
  edges = np.concatenate([truck_edges, virtual_edges])
  senders = np.concatenate([np.array(truck_senders), virtual_senders])
  receivers = np.concatenate([np.array(truck_receivers), virtual_receivers])
  return (
      Graph(nodes, edges, receivers, senders),
      network.to_undirected(),
      distances,
  )


def make_random_parcels(
    rng: np.random.Generator,
    state: Graph,
    network: nx.Graph,
    distances: np.ndarray,
    num_parcels: int,
    mean_route_length: int,
    min_parcel_weight: float = 0.01,
    max_parcel_weight: float = 1,
    parcel_weight_shape: float = 0.1,
    unit_weights: bool = False,
    start_inv_temp: float = 0.1,
    dist_inv_temp: float = 0.1,
    max_tries: int = 50,
    cut_capacities: float = 0,
) -> tuple[Graph, SolutionDict]:
  """Adds random parcels to a time-expanded transportation network.

  The network may be created using `make_random_network`, whose outputs can be
  passed as `state, network, distances`. This function adds parcels to the
  network by first sampling a parcel weight (from a truncated Pareto
  distribution [1]), a start node from the time expanded network, and then
  sampling trucks to make a parcel route. Routes end probabilistically after
  each truck to give an average route length of `mean_route_length`. The
  time-expanded network (`state`) should normally be pruned (using
  `prune_network` with `prune_parcels=False`) before adding parcels with this
  function. This method ensures that the network is completely solvable.

  Args:
    rng: Random number generator.
    state: Time-expanded transportation network in the format as generated by
      `make_random_network`. Parcels are added to a copy of this network, which
      is returned.
    network: Non-expanded (NetworkX) network corresponding to `state` (also
      returned by `make_random_network`).
    distances: Resistance distance matrix of `network`, as returned by
      `make_random_network`. This is used for sampling trucks in such a way that
      they are more likely to bring the parcel further from their start hub. How
      much this is taken into account is conrolled by the `dist_inv_temp`
      parameter.
    num_parcels: Number of parcels to be included in the network.
    mean_route_length: Average length of a parcel route (in time steps). After
      each time step of a parcel route, it is randomly decided whether to
      terminate the route, with probability 1 / `mean_route_length`. This way, a
      parcel route lasts `sum((1 - 1 / mean_route_length)**k)` time steps in
      expectation. This geometric sum approaches `mean_route_length` as the
      number of time steps in `state` becomes larger.
    min_parcel_weight: The scale (`m`) parameter of the Pareto distribution [1]
      from which parcel weights are sampled.
    max_parcel_weight: The maximum weight of parcels. The parcel weights are
      sampled from a truncated Pareto distribution using rejection sampling to
      keep weights below `max_parcel_weight`.
    parcel_weight_shape: The shape (`a`) parameter of the Pareto distribution
      [1] from which parcel weights are sampled.
    unit_weights: If `True`, all parcel weights are set to 1 instead of being
      sampled from a Pareto distribution.
    start_inv_temp: The start node hub is sampled from a Boltzmann distribution
      with exponent `-start_inv_temp * degree(hub)` to preferably start at hubs
      with lower degree.
    dist_inv_temp: When sampling parcel routes it is often posssible to choose
      between multiple trucks. In these cases, the truck is chosen among the
      available trucks by sampling from a Boltzmann distribution with exponent
      `dist_inv_temp * dist_from_start` (where the distance to the start node is
      measured in terms of resistance distance, i.e. `distances`). This
      encourages parcels to go further from their start nodes.
    max_tries: If, after a parcel route has been sampled, this parcel route
      doesn't actually use any real trucks (instead, the parcel just stays at
      its start hub), we try again. This is done for a maximum of `max_tries`
      times (after which a 'staying' parcel route will simply be accepted). The
      parcel weight is reduced by 10% each try to make it easier to place the
      parcel.
    cut_capacities: After all parcels have been placed in the network, there is
      normally still capacity left in some trucks. `cut_capacities` is a number
      between 0 and 1 that specifies how much of this excess capacity should be
      removed (by reducing the trucks' capacities). Thus, 1 means all excess
      capacity is cut and 0 means no capacity is cut.

  Returns:
    The new state (time-expanded network), now with `num_parcels` parcels added,
    as well a dictionary mapping each parcel id to a list of nodes (in
    (location, time) format) representing a solution route for that parcel.

  References:
    [1]: NumPy reference page for the Pareto distribution:
      https://numpy.org/doc/stable/reference/random/generated/numpy.random.pareto.html
  """
  timesteps = state.nodes[:, 1].max()

  # Add parcels to network.
  first_id = state.edges[:, EdgeFeatures.ID].max() + 1
  parcel_edges = np.zeros((2 * num_parcels, len(EdgeFeatures)))
  parcel_edges[::2, EdgeFeatures.ID] = first_id + np.arange(num_parcels)
  parcel_edges[1::2, EdgeFeatures.ID] = first_id + np.arange(num_parcels)
  parcel_edges[::2, EdgeFeatures.PARCEL_FORWARD] = 1
  parcel_edges[1::2, EdgeFeatures.PARCEL_BACKWARD] = 1

  if unit_weights:
    # Set all parcel weights to 1.
    weights = np.ones(num_parcels)
  else:
    # Sample parcel weights from a Pareto (power-law) distribution.
    weights = np.zeros(num_parcels)
    for i in range(num_parcels):
      while not min_parcel_weight <= weights[i] <= max_parcel_weight:
        weights[i] = (rng.pareto(parcel_weight_shape) + 1) * min_parcel_weight
    weights = np.sort(weights)[::-1]  # Place heavy parcels first.

  # Sample routes for all parcels.
  capacities = state.edges[:, EdgeFeatures.TRUCK_CAPACITY].copy()
  parcel_senders = []
  parcel_receivers = []
  solutions = {}
  for i in range(num_parcels):
    node = node_init = route = capacities_try = None
    for _ in range(max_tries):
      capacities_try = capacities.copy()

      # Sample initial node.
      loc_init = rng.choice(
          len(network.nodes),
          p=special.softmax(
              [-start_inv_temp * network.degree[node] for node in network.nodes]
          ),
      )
      node = node_init = rng.choice(
          np.where(
              (state.nodes[:, 0] == loc_init)
              & (state.nodes[:, 1] <= timesteps - mean_route_length)
          )[0]
      )

      # Sample rest of route.
      route = [tuple(state.nodes[node].astype(int))]
      while True:
        # Get available trucks from current node.
        trucks = np.where(
            (state.senders == node)  # Start at location.
            & (
                (
                    state.edges[:, EdgeFeatures.TRUCK_FORWARD] == 1
                )  # Real trucks.
                & (capacities_try >= weights[i])  # Parcel fits into truck.
                | (state.edges[:, EdgeFeatures.VIRTUAL_FORWARD] == 1)
            )
        )[0]

        if trucks.size == 0:
          break  # Dead end reached.

        # Select truck.
        dist_from_start = distances[
            loc_init, state.nodes[state.receivers[trucks], 0]
        ]
        truck = rng.choice(
            trucks, p=special.softmax(dist_inv_temp * dist_from_start)
        )
        reverse_truck = np.where(
            state.edges[:, EdgeFeatures.ID]
            == state.edges[truck, EdgeFeatures.ID]
        )

        # Add next node to route.
        node = state.receivers[truck]
        capacities_try[truck] -= weights[i]
        capacities_try[reverse_truck] -= weights[i]
        route.append(tuple(state.nodes[node].astype(int)))

        # Randomly terminate route.
        truck_steps = route[-1][1] - route[-2][1]
        if rng.binomial(truck_steps, 1 / mean_route_length) > 0:
          break

      # Check if parcel changes location. If not, try again.
      if state.nodes[node, 0] != loc_init:
        break
      else:
        weights[i] *= 0.9  # Make it a bit easier to place this parcel.

    # Add parcel goal to graph, route to solutions.
    parcel_senders.append(node_init)
    parcel_receivers.append(node)
    parcel_senders.append(node)
    parcel_receivers.append(node_init)
    assert route is not None
    solutions[int(parcel_edges[2 * i, EdgeFeatures.ID])] = route
    capacities = capacities_try

  # Weights might change during route sampling, so add to graph now.
  parcel_edges[::2, EdgeFeatures.PARCEL_WEIGHT] = weights
  parcel_edges[1::2, EdgeFeatures.PARCEL_WEIGHT] = weights

  # Cut excess truck capacities.
  state.edges[:, EdgeFeatures.TRUCK_CAPACITY] -= cut_capacities * capacities

  # Build Graph representation of network.
  edges = np.concatenate([state.edges, parcel_edges])
  senders = np.concatenate([state.senders, np.asarray(parcel_senders)])
  receivers = np.concatenate([state.receivers, np.asarray(parcel_receivers)])
  state = Graph(state.nodes, edges, receivers, senders)
  return state, solutions


def prune_parcel(network: Graph, parcel: int) -> tuple[np.ndarray, np.ndarray]:
  """Prunes `network` to only keep the parts relevant to `parcel`.

  Checks which nodes and trucks are reachable from the start node (forward in
  time) and from the goal node (backward in time). The final network consists of
  the union of reachable nodes and trucks for the given parcel.

  Args:
    network: Time-expanded transportation network in the format as generated by
      `make_random_network` with parcels as added by `make_random_parcels`.
    parcel: The forward-edge index of the parcel in `network`.

  Returns:
    A pair `(nodes, edges)`, where `nodes` are the indices of the nodes in
    `network.nodes` to remain in graph and `edges` are the indices of the edges
    in `network.edges` to remain in graph.
  """
  # Get start and end times for parcel.
  t_top = t_start = network.nodes[network.senders[parcel]][1].item()
  t_bot = t_end = network.nodes[network.receivers[parcel]][1].item()

  # Dictionaries containing nodes reachable from the top (start hub, start
  # time) and bottom (goal hub, end time), respectively.
  reachable_top = {time: set() for time in range(t_top, t_bot + 1)}
  reachable_bot = {time: set() for time in range(t_top, t_bot + 1)}
  reachable_top[t_top] = {network.senders[parcel].item()}
  reachable_bot[t_bot] = {network.receivers[parcel].item()}

  # Dictionaries containing edges encountered from the top and bottom,
  # respectively.
  edges_top = {time: [] for time in range(t_top + 1, t_bot + 1)}
  edges_bot = {time: [] for time in range(t_top, t_bot)}

  # Start at top and bottom simultaneously and move down from top while moving
  # up from the bottom. Along the way, save nodes and edges encountered. This
  # is more efficient than only going down from the top and finding paths that
  # lead to the goal.
  while t_top < t_bot:
    for node in reachable_top[t_top]:
      # Expand top node: go through possible trucks and next nodes.
      trucks_from_node = np.where(
          (network.senders == node)  # Start at location.
          & (
              network.nodes[network.receivers][:, 1] <= t_end
          )  # Don't overshoot goal.
          & (
              (network.edges[:, EdgeFeatures.TRUCK_FORWARD] == 1)
              & (
                  network.edges[:, EdgeFeatures.TRUCK_CAPACITY]
                  >= network.edges[parcel, EdgeFeatures.PARCEL_WEIGHT]
              )
              | (network.edges[:, EdgeFeatures.VIRTUAL_FORWARD] == 1)
          )  # Only consider real and virtual trucks with enough capacity.
      )[0]
      for truck in trucks_from_node:
        # Save all reachable trucks and nodes from here.
        hub = network.receivers[truck]
        reachable_top[network.nodes[hub][1].item()].add(hub.item())
        edges_top[network.nodes[hub][1].item()].append(truck.item())
    t_top += 1

    if t_bot == t_top:
      break  # If bottom time = top time, we are done.

    for node in reachable_bot[t_bot]:
      # Expand bottom node: go through possible trucks and previous nodes.
      trucks_to_node = np.where(
          (network.receivers == node)  # End at location.
          & (
              network.nodes[network.senders][:, 1] >= t_start
          )  # Don't overshoot start.
          & (
              (network.edges[:, EdgeFeatures.TRUCK_FORWARD] == 1)
              & (
                  network.edges[:, EdgeFeatures.TRUCK_CAPACITY]
                  >= network.edges[parcel, EdgeFeatures.PARCEL_WEIGHT]
              )
              | (network.edges[:, EdgeFeatures.VIRTUAL_FORWARD] == 1)
          )  # Only consider real and virtual trucks with enough capacity.
      )[0]
      for truck in trucks_to_node:
        # Save all reachable trucks and nodes from here.
        hub = network.senders[truck]
        reachable_bot[network.nodes[hub][1].item()].add(hub.item())
        edges_bot[network.nodes[hub][1].item()].append(truck.item())
    t_bot -= 1

  # Of all the reachable nodes and edges we now want to keep only those that
  # are reachable both from the top and from the bottom. To find these, we go
  # once through the top-reachable edges, starting at the bottom and going up
  # to the top. Along the way, we remove edges not reachable from the bottom
  # and mark reachable nodes as reachable from the bottom. Similarly we will
  # go down from the top to the bottom to do the same for top-reachability.
  # Go up to start from the goal to check bottom-reachability.
  for time in range(t_end, t_start, -1):
    if not edges_top[time]:
      continue
    # Remove unreachable edges.
    edges_top[time] = np.array(edges_top[time])
    edges_top[time] = edges_top[time][
        np.isin(
            network.receivers[edges_top[time]],
            np.array(list(reachable_bot[time])),
        )
    ]
    # Mark reachable nodes.
    for truck in edges_top[time]:
      hub = network.senders[truck]
      reachable_bot[network.nodes[hub][1].item()].add(hub.item())

  # Go down to goal from the start to check top-reachability.
  for time in range(t_start, t_end):
    if not edges_bot[time]:
      continue
    # Remove unreachable edges.
    edges_bot[time] = np.array(edges_bot[time])
    edges_bot[time] = edges_bot[time][
        np.isin(
            network.senders[edges_bot[time]],
            np.array(list(reachable_top[time])),
        )
    ]
    # Mark reachable nodes.
    for truck in edges_bot[time]:
      hub = network.receivers[truck]
      reachable_top[network.nodes[hub][1].item()].add(hub.item())

  # Unify top- and bottom-reachable nodes and trucks.
  reachable = []
  if t_start in reachable_top and t_start in reachable_bot:
    reachable.extend(reachable_top[t_start] & reachable_bot[t_start])
  trucks = set()
  for time in range(t_start + 1, t_end + 1):
    reachable.extend(reachable_top[time] & reachable_bot[time])
    trucks.update(np.asarray(edges_top[time]).tolist())
    trucks.update(np.asarray(edges_bot[time - 1]).tolist())
  trucks = np.array(list(trucks))

  # Add reverse truck edges and parcel edges.
  if trucks.size > 0:
    trucks = np.where(
        network.edges[:, EdgeFeatures.ID]
        == network.edges[trucks, EdgeFeatures.ID][:, None]
    )[1]
  parcel = np.where(
      network.edges[:, EdgeFeatures.ID]
      == network.edges[parcel, EdgeFeatures.ID]
  )[0]

  edges = np.unique(np.concatenate([trucks, parcel])).astype(int)
  nodes = np.unique(np.array(reachable)).astype(int)
  return nodes, edges


def prune_network(
    network: Graph,
    prune_parcels: bool = True,
    solution: Optional[SolutionDict] = None,
) -> tuple[
    Graph,
    Mapping[tuple[tuple[int, int], tuple[int, int]], Sequence[tuple[int, int]]],
    Optional[Mapping[int, set[int]]],
    Optional[SolutionDict],
]:
  """Removes nodes and edges from network that are not relevant.

  If `prune_parcels` is `True`, then this function checks, or every parcel in
  the network, which nodes and trucks are reachable from the start node (forward
  in time) and from the goal node (backward in time). The final network consists
  of the union of reachable nodes and trucks for all parcels.

  Regardless of `prune_parcels`, nodes which have only one incoming and one
  outgoing connection (and are neither the start nor the end node for a parcel)
  are removed from the network, and edges to skip these nodes are inserted.

  Args:
    network: Time-expanded transportation network ('state') in the format as
      generated by `make_random_network` with parcels as added by
      `make_random_parcels`. Not modified in-place.
    prune_parcels: Whether to only keep nodes and edges relevant to the parcels
      in the network (`True`), or to ignore parcels and only remove skippable
      nodes (`False`).
    solution: If provided, the node ids in these routes are modified to
      correspond to those in the pruned network. Not modified in-place.

  Returns:
    - A new logistics network where unreachable trucks and nodes as well as
      skippable nodes have been removed from the original one.
    - A dictionary containing a list of all skipped nodes for each newly
      inserted edge.
    - A dictionary containing, for each unpruned edge, a list of all parcels
      which include this edge in their pruned graph. This is important in
      feature graph construction to understand which parcels from outside the
      feature graph might affect truck capacities inside the feature graph.
      If `prune_parcels == False`, then this is `None`.
    - If `solution` is provided, a corresponding new `solution` dict is also
      returned, otherwise this is `None`.
  """
  edge_parcels = new_solution = None
  if prune_parcels:
    # Find all relevant nodes and edges for each parcel.
    parcels = np.where(network.edges[:, EdgeFeatures.PARCEL_FORWARD])[0]
    nodes = []
    edges = []
    # Record which parcels care about each edge.
    edge_parcels = collections.defaultdict(set)
    for parcel in parcels:
      parcel_nodes, parcel_edges = prune_parcel(network, parcel)
      nodes.append(parcel_nodes)
      edges.append(parcel_edges)
      for edge in parcel_edges:
        edge_parcels[int(network.edges[edge, EdgeFeatures.ID])].add(
            int(network.edges[parcel, EdgeFeatures.ID])
        )

    nodes = np.unique(np.concatenate(nodes))
    edges = np.unique(np.concatenate(edges))
  else:
    nodes = np.arange(len(network.nodes))
    edges = np.arange(len(network.edges))

  # Now we remove all nodes which are skippable (have only one incoming and one
  # outgoing edge). These nodes don't require decision-making, so they don't
  # need to be part of the state.
  # First, identify all skippable nodes.
  forward_connections = edges[
      (network.edges[edges, EdgeFeatures.TRUCK_FORWARD] == 1)
      | (network.edges[edges, EdgeFeatures.VIRTUAL_FORWARD] == 1)
  ]
  parcel_connections = edges[
      (network.edges[edges, EdgeFeatures.PARCEL_FORWARD] == 1)
      | (network.edges[edges, EdgeFeatures.PARCEL_BACKWARD] == 1)
  ]
  skippable_nodes = nodes[
      (
          (network.receivers[forward_connections] == nodes[:, None]).sum(1) == 1
      )  # Exactly one incoming forward connection.
      & (
          (network.senders[forward_connections] == nodes[:, None]).sum(1) == 1
      )  # Exactly one outgoing forward connection.
      & (
          (network.senders[parcel_connections] == nodes[:, None]).sum(1) == 0
      )  # Not a parcel start or goal node.
  ]

  # Now, build new connections to skip the skippable nodes. The capacity of a
  # connection is the capacity of the smallest truck that is skipped.
  connections = []  # List of (start node, end node, truck capacity).
  connection_idx = {}  # Skipped node to connection index assignments.
  for skippable_node in skippable_nodes:
    incoming = forward_connections[
        np.where(network.receivers[forward_connections] == skippable_node)[0][0]
    ]
    outgoing = forward_connections[
        np.where(network.senders[forward_connections] == skippable_node)[0][0]
    ]
    above = network.senders[incoming]
    below = network.receivers[outgoing]

    if above in connection_idx:
      # Node above already has a connection associated. Add this node to the
      # same connection, making sure to update the end node to the one below and
      # the capacity if necessary.
      above_idx = connection_idx[above]
      connection_idx[skippable_node] = above_idx
      connections[above_idx][1] = below
      capacity = (
          network.edges[incoming, EdgeFeatures.TRUCK_CAPACITY]
          if network.edges[incoming, EdgeFeatures.TRUCK_FORWARD] == 1
          else np.inf
      )
      connections[above_idx][2] = min(connections[above_idx][2], capacity)

      if below in connection_idx:
        # Node below also has a connection associated. Unite both connections.
        below_idx = connection_idx[below]
        for node in connection_idx:
          if connection_idx[node] == below_idx:
            connection_idx[node] = above_idx
        connections[above_idx][1] = connections[below_idx][1]
        connections[above_idx][2] = min(
            connections[above_idx][2], connections[below_idx][2]
        )

    elif below in connection_idx:
      # Node below already has a connection associated. Add this node to the
      # same connection, making sure to update the end node to the one below and
      # the capacity if necessary.
      below_idx = connection_idx[below]
      connection_idx[skippable_node] = below_idx
      connections[below_idx][0] = above
      capacity = (
          network.edges[outgoing, EdgeFeatures.TRUCK_CAPACITY]
          if network.edges[outgoing, EdgeFeatures.TRUCK_FORWARD] == 1
          else np.inf
      )
      connections[below_idx][2] = min(
          connections[below_idx][2],
          capacity,
      )

    else:
      # Nodes above and below are not part of any connection, make a new one to
      # skip this node.
      capacity_incoming = (
          network.edges[incoming, EdgeFeatures.TRUCK_CAPACITY]
          if network.edges[incoming, EdgeFeatures.TRUCK_FORWARD] == 1
          else np.inf
      )
      capacity_outgoing = (
          network.edges[outgoing, EdgeFeatures.TRUCK_CAPACITY]
          if network.edges[outgoing, EdgeFeatures.TRUCK_FORWARD] == 1
          else np.inf
      )
      connections.append(
          [above, below, min(capacity_incoming, capacity_outgoing)]
      )
      connection_idx[skippable_node] = len(connections) - 1

  # Build a dictionary containing the information about which skipped nodes are
  # part of which new connection. This is important for real-world actuation,
  # because the new connections are not part of the original network, so parcels
  # cannot actually be put onto these "trucks".
  # Key = (connection start, connection end), value = list of skipped nodes.
  connections_info = {}
  used_connections = set(connection_idx.values())
  for i in used_connections:
    connections_info[(
        tuple(network.nodes[connections[i][0]]),
        tuple(network.nodes[connections[i][1]]),
    )] = sorted(
        [
            tuple(network.nodes[node])
            for node in skippable_nodes
            if connection_idx[node] == i
        ],
        key=lambda node: node[1],
    )

  # Now actually remove the skippable nodes and associated edges from the
  # network.
  # Delete unused connections.
  connections = [connections[i] for i in used_connections]
  nodes = nodes[~np.isin(nodes, skippable_nodes)]
  edges = edges[
      ~np.isin(network.senders[edges], skippable_nodes)
      & ~np.isin(network.receivers[edges], skippable_nodes)
  ]

  # Add new edges according to the specifications in `connections`.
  new_edges = np.zeros((2 * len(connections), len(EdgeFeatures)))
  new_receivers = np.zeros((2 * len(connections)), int)
  new_senders = np.zeros((2 * len(connections)), int)
  start_id = network.edges[:, EdgeFeatures.ID].max() + 1
  for i, connection in enumerate(connections):
    new_senders[2 * i], new_senders[2 * i + 1], _ = connection
    new_receivers[2 * i + 1], new_receivers[2 * i], _ = connection
    new_edges[[2 * i, 2 * i + 1], EdgeFeatures.ID] = start_id + i
    if np.isfinite(connection[2]):
      new_edges[2 * i, EdgeFeatures.TRUCK_FORWARD] = 1
      new_edges[2 * i + 1, EdgeFeatures.TRUCK_BACKWARD] = 1
      new_edges[[2 * i, 2 * i + 1], EdgeFeatures.TRUCK_CAPACITY] = connection[2]
    else:
      # A connection has infinite capacity if all skipped edges were virtual.
      new_edges[2 * i, EdgeFeatures.VIRTUAL_FORWARD] = 1
      new_edges[2 * i + 1, EdgeFeatures.VIRTUAL_BACKWARD] = 1

  # Build Graph representation.
  receivers = np.concatenate([network.receivers[edges], new_receivers])
  senders = np.concatenate([network.senders[edges], new_senders])
  receivers = np.where(nodes == receivers[:, None])[1]
  senders = np.where(nodes == senders[:, None])[1]
  network = Graph(
      network.nodes[nodes],
      np.concatenate([network.edges[edges], new_edges]),
      receivers,
      senders,
  )

  if solution is not None:
    # Build new solution dict by removing skipped route parts.
    new_solution = {}
    for parcel, route in solution.items():
      new_solution[parcel] = []
      skip_to = 0
      for i, first_stop in enumerate(route):
        if i < skip_to:
          continue
        new_solution[parcel].append(first_stop)
        for j, second_stop in enumerate(route[i + 2 :]):
          if (first_stop, second_stop) in connections_info:
            # This part of the route has been removed.
            skip_to = j + i + 2
            break
  return network, connections_info, edge_parcels, new_solution


def clean_step(
    state: Graph,
    parcel: int,
    prev_node: int,
    remove_parcel: bool,
    prune: bool = True,
) -> tuple[Graph, Optional[int]]:
  """Prune state after a parcel has been moved.

  This function is more efficient than calling `prune_network`, because it only
  considers how the state should be pruned as a result of (re)moving `parcel`.

  Args:
    state: Time-expanded transportation network with parcels.
    parcel: Index of the forward-in-time parcel edge in `state`.
    prev_node: Index of the previous sender node of `parcel` in `state`. If the
      parcel has not been moved this can also be the current sender of `parcel`.
    remove_parcel: Whether to remove the parcel from `state`.
    prune: Whether to do the pruning step. When disabled, this function can
      still be used to remove a parcel from the network. If both `prune` and
      `remove_parcel` are `False`, this function does not do anything.

  Returns:
    The newly cleaned state and the (possibly changed) index of `parcel` in the
    new state. If the parcel has been removed, then this index is `None`.
  """
  if not prune and not remove_parcel:
    return state, parcel  # Nothing to do.

  keep_edge = np.ones(len(state.edges), bool)
  nodes_set = set(range(len(state.nodes)))

  reverse_parcel = np.where(
      (state.edges[:, EdgeFeatures.PARCEL_BACKWARD] == 1)
      & (
          state.edges[:, EdgeFeatures.ID]
          == state.edges[parcel, EdgeFeatures.ID]
      )
  )[0][0]

  keep_edge[[parcel, reverse_parcel]] = not remove_parcel

  # Remove unreachable nodes and trucks from graph.
  open_nodes = [prev_node]
  while open_nodes and prune:
    node = open_nodes.pop()
    if node not in nodes_set:
      # Node already removed.
      continue

    # Skip node if it is reachable.
    if (
        state.edges[(state.receivers == node) & keep_edge][
            :,
            [
                EdgeFeatures.PARCEL_FORWARD,
                EdgeFeatures.PARCEL_BACKWARD,
                EdgeFeatures.TRUCK_FORWARD,
                EdgeFeatures.VIRTUAL_FORWARD,
            ],
        ].sum()
        > 0
    ):
      continue

    # Remove node from nodes, remove outgoing (bidirectional) edges, add child
    # nodes to open.
    nodes_set.remove(node)
    trucks_from_node = (
        (state.senders == node)
        & (
            state.edges[
                :,
                [
                    EdgeFeatures.TRUCK_FORWARD,
                    EdgeFeatures.VIRTUAL_FORWARD,
                ],
            ].sum()
            > 0
        )
    ) | (
        (state.receivers == node)
        & (
            state.edges[
                :,
                [
                    EdgeFeatures.TRUCK_BACKWARD,
                    EdgeFeatures.VIRTUAL_BACKWARD,
                ],
            ].sum()
            > 0
        )
    )
    children = set(state.receivers[keep_edge & trucks_from_node].tolist()) - {
        node
    }
    open_nodes.extend(children)
    keep_edge = keep_edge & ~trucks_from_node

  # Build Graph representation of the new state.
  nodes = np.fromiter(nodes_set, dtype=int)
  receivers = np.where(nodes == state.receivers[keep_edge][:, None])[1]
  senders = np.where(nodes == state.senders[keep_edge][:, None])[1]
  parcel = np.sum(keep_edge[:parcel]) if not remove_parcel else None
  return (
      Graph(state.nodes[nodes], state.edges[keep_edge], receivers, senders),
      parcel,
  )


def draw_network(
    network: Graph,
    ax: plt.Axes,
    solution: Optional[SolutionDict] = None,
    node_feature: Optional[int] = None,
    time_axis: bool = True,
    index_labels: bool = False,
    draw_kwargs: Optional[Mapping[str, Any]] = None,
) -> None:
  """Draw time-expanded transportation network `network` onto `ax`.

  Draws real and virtual truck edges that go forward in time. Draws parcels by
  coloring in the start and goal nodes in the same color. If given a solution
  (list of routes), then these routes are plotted by highlighting the trucks
  that are part of the solution in the same color as the corresponding parcel.
  Otherwise, draws the parcel edges (from start to goal).

  Note that ambiguity can arise in the visualization if several parcels share
  the same locations or trucks.

  Args:
    network: Time-expanded transportation network in the format as generated by
      `make_random_network`, either with or without parcels.
    ax: Matplotlib `Axes` object onto which the graph is plotted.
    solution: If provided, these routes are highlighted in the same color as the
      corresponding parcel. Must be of the list-of-routes format as returned by
      `make_random_parcels` with `return_solution=True`.
    node_feature: Index of node feature which shall be used to color in nodes.
      If `None`, nodes are grey (if without parcel) or the color assigned to the
      parcel.
    time_axis: Whether to make time (second node feature) go downwards in the
      plot. Might be turned off to reduce ambiguity with edges which otherwise
      lie on top of each other.
    index_labels: Whether to use a node's index in the network as its label
      instead of the hub index (first node feature).
  """
  # Turn Graph representation into NetworkX representation to plot.
  # TODO(onno): Maybe move reusable part of `draw_network` to `Graph`
  nx_graph = nx.DiGraph()
  node_colors = {}
  edge_colors = {}
  node_edgecolors = {}
  parcel_colors = {}
  for i, node in enumerate(network.nodes):
    nx_graph.add_node(i, time=int(node[1]))
    if node_feature is not None:
      node_colors[i] = node[node_feature]
  for i, edge in enumerate(network.edges):
    pair = (int(network.senders[i]), int(network.receivers[i]))
    if edge[EdgeFeatures.TRUCK_FORWARD] or edge[EdgeFeatures.VIRTUAL_FORWARD]:
      # Edge represents real or virtual truck.
      nx_graph.add_edge(*pair)
      if solution is not None:
        # Color in edge if part of a solution.
        for parcel, route in solution.items():
          for stop, next_stop in zip(route, route[1:]):
            if stop == tuple(
                network.nodes[pair[0], :2].astype(int)
            ) and next_stop == tuple(network.nodes[pair[1], :2].astype(int)):
              if parcel not in parcel_colors:
                parcel_colors[parcel] = f'C{len(parcel_colors)}'
              color = parcel_colors[parcel]
              edge_colors[pair] = color
              node_edgecolors[pair[0]] = color
              node_edgecolors[pair[1]] = color

    elif edge[EdgeFeatures.PARCEL_FORWARD]:
      # Edge represents parcel.
      parcel = int(edge[EdgeFeatures.ID])
      if parcel not in parcel_colors:
        parcel_colors[parcel] = f'C{len(parcel_colors)}'
      color = parcel_colors[parcel]
      if node_feature is None:
        node_colors[pair[0]] = color
        node_colors[pair[1]] = color
      if solution is None:
        nx_graph.add_edge(*pair)
        edge_colors[pair] = color
        node_edgecolors[pair[0]] = color
        node_edgecolors[pair[1]] = color

  # Plot network.
  pos = nx.multipartite_layout(
      nx_graph, subset_key='time', align='horizontal', scale=-10
  )
  draw_kwargs = {'font_color': 'k', 'node_size': 500} | dict(draw_kwargs or {})
  nx.draw(
      nx_graph,
      pos=pos if time_axis else None,
      labels={
          node: node if index_labels else int(network.nodes[node][0])
          for node in nx_graph.nodes
      },
      node_color=[
          node_colors.get(node, 'lightgrey') for node in nx_graph.nodes
      ],
      edge_color=[
          edge_colors.get(edge, 'lightgrey' if parcel_colors else 'grey')
          for edge in nx_graph.edges
      ],
      ax=ax,
      edgecolors=[node_edgecolors.get(node, 'k') for node in nx_graph.nodes],
      **draw_kwargs,
  )
