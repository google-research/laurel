"""Functions for working with graphs."""

from collections.abc import Sequence
import enum

import jax
import jax.numpy as jnp
import jraph
import matplotlib.pyplot as plt
import networkx as nx


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


def make_random_network(
    key: jax.random.PRNGKeyArray,
    num_hubs: int,
    timesteps: int,
    num_trucks_per_step: int,
    max_truck_duration: int,
    max_truck_capacity: float,
    network_connectivity: int = 2,
) -> jraph.GraphsTuple:
  """Generates a random time-expanded transportation network.

  First, generates a network of depots and connections between them according to
  the Barabási-Albert algorithm. This results in a few highly connected depots
  (large degree), while most depots only have a few connections. The
  time-expanded representation is then generated from the original network by
  sampling a pre-specified number of trucks at each time step, each of which
  starts at the current time step and ends a random number of time steps later
  (taking care to not exceed the specified total number of time steps). Parcels
  may be added with the `make_random_parcels` function.

  Args:
    key: Source of pseudorandomness.
    num_hubs: Number of depots in the network.
    timesteps: Number of time steps.
    num_trucks_per_step: Number of trucks leaving every time step
      (deterministic).
    max_truck_duration: The maximal number of time steps a truck takes to reach
      its destination. The actual number is sampled uniformly from {1, ...,
      `max_truck_duration`}.
    max_truck_capacity: The maximum scalar capacity of trucks. The actual
      capacity is sampled uniformly from [0, `max_truck_capacity`).
    network_connectivity: The `m` parameter of the Barabási-Albert model.

  Returns:
    The time-expanded network representation as a `jraph.GraphsTuple`. The node
    features are tuples of the form (location, time). The edge features are of
    the form (edge id, forward truck?, backward truck?, forward parcel?,
    backward parcel?, forward virtual?, backward virtual?, capacity if truck,
    weight if parcel).
  """
  # Generate transportation network.
  key, subkey = jax.random.split(key)
  network = nx.barabasi_albert_graph(
      num_hubs, network_connectivity, int(subkey[1])
  ).to_directed()

  # Node features for time-expanded network: (location, time).
  nodes = jnp.hstack([
      jnp.tile(jnp.arange(num_hubs), timesteps + 1)[:, None],
      jnp.repeat(jnp.arange(timesteps + 1), num_hubs)[:, None],
  ])

  # Trucks for time-expanded network (sample one time step at a time).
  truck_senders = []
  truck_receivers = []
  num_trucks = 0
  for timestep in range(timesteps):
    # Sample trucks and add to network.
    key, *subkeys = jax.random.split(key, 3)
    trucks = jax.random.choice(
        subkeys[0],
        jnp.asarray(network.edges),
        (num_trucks_per_step,),
        replace=False,
    )
    times = 1 + jax.random.choice(
        subkeys[1], max_truck_duration, (num_trucks_per_step,)
    )
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
  capacities = max_truck_capacity * jax.random.uniform(key, (num_trucks,))
  truck_edges = (
      jnp.zeros((2 * num_trucks, len(EdgeFeatures)))
      .at[::2, EdgeFeatures.ID]
      .set(jnp.arange(num_trucks))
      .at[1::2, EdgeFeatures.ID]
      .set(jnp.arange(num_trucks))
      .at[::2, EdgeFeatures.TRUCK_FORWARD]
      .set(1)
      .at[1::2, EdgeFeatures.TRUCK_BACKWARD]
      .set(1)
      .at[::2, EdgeFeatures.TRUCK_CAPACITY]
      .set(capacities)
      .at[1::2, EdgeFeatures.TRUCK_CAPACITY]
      .set(capacities)
  )

  # Add virtual trucks ("stay at location") to network.
  virtual_edges = (
      jnp.zeros((2 * timesteps * num_hubs, len(EdgeFeatures)))
      .at[::2, EdgeFeatures.ID]
      .set(jnp.arange(timesteps * num_hubs))
      .at[1::2, EdgeFeatures.ID]
      .set(jnp.arange(timesteps * num_hubs))
      .at[::2, EdgeFeatures.VIRTUAL_FORWARD]
      .set(1)
      .at[1::2, EdgeFeatures.VIRTUAL_BACKWARD]
      .set(1)
  )
  virtual_senders = (
      jnp.repeat(jnp.arange(num_hubs * timesteps), 2).at[1::2].add(num_hubs)
  )
  virtual_receivers = (
      jnp.repeat(jnp.arange(num_hubs * timesteps), 2).at[::2].add(num_hubs)
  )

  # Build Jraph representation of network.
  edges = jnp.concatenate([truck_edges, virtual_edges])
  senders = jnp.concatenate([jnp.array(truck_senders), virtual_senders])
  receivers = jnp.concatenate([jnp.array(truck_receivers), virtual_receivers])
  return jraph.GraphsTuple(
      nodes=nodes,
      edges=edges,
      receivers=receivers,
      senders=senders,
      globals=None,
      n_node=jnp.array([len(nodes)]),
      n_edge=jnp.array([len(edges)]),
  )


def make_random_parcels(
    key: jax.random.PRNGKeyArray,
    network: jraph.GraphsTuple,
    num_hubs: int,
    num_parcels: int,
    max_route_length: int,
    max_parcel_weight: float,
    virtual_truck_rate: float = 0.2,
    return_solution: bool = False,
) -> (
    jraph.GraphsTuple
    | tuple[jraph.GraphsTuple, Sequence[Sequence[tuple[int, int]]]]
):
  """Adds random parcels to a time-expanded transportation network.

  The network may be created using `make_random_network` and is passed as
  `network`. Parcels are added to the network by first sampling the route
  length, then selecting a suitable start node, and finally picking a goal by
  sampling a suitable truck route from the start node. The method ensures that
  the network is completely solvable.

  Args:
    key: Source of pseudorandomness.
    network: Time-expanded transportation network in the format as generated by
      `make_random_network`. Parcels are added to a copy of this network, which
      is returned.
    num_hubs: Number of depots in `network`.
    num_parcels: Number of parcels to be included in the network.
    max_route_length: Maximum number of time steps that parcel routes have. The
      number of time steps that a route has is sampled uniformly from {1, ...,
      `max_route_length`}.
    max_parcel_weight: The maximum scalar weight of parcels. The actual weight
      is sampled uniformly from [0, `max_parcel_weight`).
    virtual_truck_rate: Positive real number specifying how likely "virtual"
      truck selection (staying at a hub) is when sampling a parcel route,
      compared to "real" truck selection. The probability of sampling a virtual
      truck is `virtual_truck_rate` times the probability of sampling a given
      real truck, assuming there are real trucks available (otherwise the
      probability of sampling a virtual truck is 1).
    return_solution: Whether to return the sampled parcel routes (which solve
      the problem). The format is a list of routes, where each route is a list
      of 2-tuples (i, j), representing an edge from node i to node j.

  Returns:
    The time-expanded network representation with parcels as a
    `jraph.GraphsTuple`. The node features (if enabled with
    `node_features = True`) are one-hot encodings of the hub locations the edge
    features are of the form (truck?, parcel?, virtual?, capacity if truck,
    weight if parcel). If `return_solution` is true, then (possibly suboptimal)
    solutions are also returned for all parcels.
  """
  # Extract network information.
  edges = jnp.asarray(network.edges)
  senders = network.senders
  receivers = network.receivers
  timesteps = network.n_node[0] // num_hubs - 1

  # Add parcels to network.
  key, subkey = jax.random.split(key)
  weights = jax.random.uniform(subkey, (num_parcels,)) * max_parcel_weight
  parcel_edges = (
      jnp.zeros((2 * num_parcels, len(EdgeFeatures)))
      .at[::2, EdgeFeatures.ID]
      .set(jnp.arange(num_parcels))
      .at[1::2, EdgeFeatures.ID]
      .set(jnp.arange(num_parcels))
      .at[::2, EdgeFeatures.PARCEL_FORWARD]
      .set(1)
      .at[1::2, EdgeFeatures.PARCEL_BACKWARD]
      .set(1)
      .at[::2, EdgeFeatures.PARCEL_WEIGHT]
      .set(weights)
      .at[1::2, EdgeFeatures.PARCEL_WEIGHT]
      .set(weights)
  )

  # Sample routes for all parcels.
  capacities = edges[:, EdgeFeatures.TRUCK_CAPACITY]
  parcel_senders = []
  parcel_receivers = []
  solutions = []
  for i in range(num_parcels):
    weight = weights[i]
    key, *subkeys = jax.random.split(key, 4)

    # Sample route length and start location/time, add start node to graph.
    length = jax.random.choice(subkeys[0], max_route_length) + 1
    time = jax.random.choice(subkeys[1], timesteps - length + 1)
    loc = jax.random.choice(subkeys[2], num_hubs)
    parcel_senders.append(loc + num_hubs * time)

    # Sample rest of route.
    node = loc + num_hubs * time
    route = []
    end = time + length  # Final time of route.
    while time < end:
      route.append(node)

      # Get available real trucks that fit into route.
      trucks = jnp.where(
          (senders == node)  # Start at location.
          & (receivers // num_hubs <= end)  # Don't overshoot goal.
          & (edges[:, EdgeFeatures.TRUCK_FORWARD] == 1)  # Real trucks.
          & (capacities >= weight)  # Parcel fits into truck.
      )[0]

      # Select truck and add to route.
      key, *subkeys = jax.random.split(key, 3)
      p_real_truck = (
          len(trucks) / (virtual_truck_rate + len(trucks))
          if len(trucks)
          else 0.0
      )
      real_truck = jax.random.bernoulli(subkeys[0], p_real_truck).item()
      if real_truck:
        truck = jax.random.choice(subkeys[1], trucks)
        time = receivers[truck] // num_hubs
        loc = receivers[truck] % num_hubs
        capacities = capacities.at[truck].add(-weight)
      else:
        time += 1

      # Update route.
      node = loc + num_hubs * time
      route[-1] = (int(route[-1]), int(node))

    # Add parcel goal to graph, route to solutions.
    parcel_receivers.append(loc + num_hubs * time)
    parcel_senders.append(parcel_receivers[-1])
    parcel_receivers.append(parcel_senders[-2])
    solutions.append(route)

  # Build Jraph representation of network.
  edges = jnp.concatenate([edges, parcel_edges])
  senders = jnp.concatenate([senders, jnp.asarray(parcel_senders)])
  receivers = jnp.concatenate([receivers, jnp.asarray(parcel_receivers)])
  network = jraph.GraphsTuple(
      nodes=network.nodes,
      edges=edges,
      receivers=receivers,
      senders=senders,
      globals=None,
      n_node=network.n_node,
      n_edge=jnp.array([len(edges)]),
  )
  if return_solution:
    return network, solutions
  return network


def prune_network(
    network: jraph.GraphsTuple,
    solution: Sequence[Sequence[tuple[int, int]]] | None = None,
) -> (
    jraph.GraphsTuple
    | tuple[jraph.GraphsTuple, Sequence[Sequence[tuple[int, int]]]]
):
  """Removes trucks from network that are not relevant for the included parcels.

  For every parcel in the network, checks which nodes and trucks are reachable
  from the start node (forward in time) and from the goal node (backward in
  time). The final network consists of the union of reachable nodes and trucks
  for all parcels.

  Args:
    network: Time-expanded transportation network in the format as generated by
      `make_random_network` with parcels as added by `make_random_parcels`. Not
      modified in-place.
    solution: If provided, the node ids in these routes are modified to
      correspond to those in the pruned network. Not modified in-place.

  Returns:
    A new logistics network where unreachable trucks and nodes have been removed
    from the original one. If `solution` is provided, a corresponding new
    `solution` list is also returned.
  """
  if not isinstance(network.nodes, jax.Array):
    raise ValueError('`network` is not a valid logistics network.')

  # Find all relevant nodes and edges for each parcel.
  parcels = jnp.where(network.edges[:, EdgeFeatures.PARCEL_FORWARD])[0]
  nodes = []
  edges = []
  for parcel in parcels:
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
        trucks_from_node = jnp.where(
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
        trucks_to_node = jnp.where(
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
      edges_top[time] = jnp.array(edges_top[time])
      edges_top[time] = edges_top[time][
          jnp.isin(
              network.receivers[edges_top[time]],
              jnp.array(list(reachable_bot[time])),
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
      edges_bot[time] = jnp.array(edges_bot[time])
      edges_bot[time] = edges_bot[time][
          jnp.isin(
              network.senders[edges_bot[time]],
              jnp.array(list(reachable_top[time])),
          )
      ]
      # Mark reachable nodes.
      for truck in edges_bot[time]:
        hub = network.receivers[truck]
        reachable_top[network.nodes[hub][1].item()].add(hub.item())

    # Unify top- and bottom-reachable nodes and trucks.
    reachable = list(reachable_top[t_start] & reachable_bot[t_start])
    trucks = set()
    for time in range(t_start + 1, t_end + 1):
      reachable.extend(reachable_top[time] & reachable_bot[time])
      trucks.update(jnp.asarray(edges_top[time]).tolist())
      trucks.update(jnp.asarray(edges_bot[time - 1]).tolist())
    trucks = jnp.array(list(trucks))

    # Add reverse truck edges and parcel edges.
    reverse_trucks = jnp.where(
        (
            network.edges[:, EdgeFeatures.TRUCK_BACKWARD]
            == network.edges[trucks, EdgeFeatures.TRUCK_FORWARD][:, None]
        )
        & (
            network.edges[:, EdgeFeatures.VIRTUAL_BACKWARD]
            == network.edges[trucks, EdgeFeatures.VIRTUAL_FORWARD][:, None]
        )
        & (
            network.edges[:, EdgeFeatures.ID]
            == network.edges[trucks, EdgeFeatures.ID][:, None]
        )
    )[1]
    reverse_parcel = jnp.where(
        (network.edges[:, EdgeFeatures.PARCEL_BACKWARD] == 1)
        & (
            network.edges[:, EdgeFeatures.ID]
            == network.edges[parcel, EdgeFeatures.ID]
        )
    )[0]
    edges += [trucks, reverse_trucks, jnp.array([parcel]), reverse_parcel]
    nodes += reachable

  # Build Jraph representation.
  edges = jnp.unique(jnp.concatenate(edges))
  nodes = jnp.unique(jnp.array(nodes))
  receivers = jnp.where(nodes == network.receivers[edges][:, None])[1]
  senders = jnp.where(nodes == network.senders[edges][:, None])[1]
  network = jraph.GraphsTuple(
      nodes=network.nodes[nodes],
      edges=network.edges[edges],
      receivers=receivers,
      senders=senders,
      globals=None,
      n_node=jnp.array([len(nodes)]),
      n_edge=jnp.array([len(edges)]),
  )

  if solution is not None:
    # Build new solution list by relabelling node ids.
    new_solution = []
    for route in solution:
      new_solution.append([])
      for a, b in route:
        new_solution[-1].append((
            jnp.where(nodes == a)[0][0].item(),
            jnp.where(nodes == b)[0][0].item(),
        ))
    return network, new_solution
  return network


def draw_network(
    network: jraph.GraphsTuple,
    ax: plt.Axes,
    solution: Sequence[Sequence[tuple[int, int]]] | None = None,
) -> None:
  """Draw time-expanded transportation network `network` onto `ax`.

  Draws real and virtual truck edges that go forward in time. Draws parcels by
  coloring in the start and goal nodes in the same color. If given a solution
  (list of routes), then these routes are plotted by highlighting the trucks
  that are part of the solution in the same color as the corresponding parcel.

  Note that ambiguity can arise in the visualization if several parcels share
  the same locations or trucks.

  Args:
    network: Time-expanded transportation network in the format as generated by
      `make_random_network`, either with or without parcels.
    ax: Matplotlib `Axes` object onto which the graph is plotted.
    solution: If provided, these routes are highlighted in the same color as the
      corresponding parcel. Must be of the list-of-routes format as returned by
      `make_random_parcels` with `return_solution=True`.
  """
  # Turn Jraph representation into NetworkX representation to plot.
  nx_graph = nx.DiGraph()
  node_colors = {}
  edge_colors = {}
  node_edgecolors = {}
  colored_nodes = 0
  for i, node in enumerate(network.nodes):
    nx_graph.add_node(i, time=int(node[1]))
  for i, edge in enumerate(network.edges):
    if edge[EdgeFeatures.TRUCK_FORWARD] or edge[EdgeFeatures.VIRTUAL_FORWARD]:
      # Edge represents real or virtual truck.
      pair = (int(network.senders[i]), int(network.receivers[i]))
      nx_graph.add_edge(*pair)
      if solution is not None:
        # Color in edge if part of a solution.
        for j, _ in enumerate(solution):
          if pair in solution[j]:
            edge_colors[pair] = f'C{j}'
            node_edgecolors[pair[0]] = f'C{j}'
            node_edgecolors[pair[1]] = f'C{j}'
    if edge[EdgeFeatures.PARCEL_FORWARD]:
      # Edge represents parcel.
      c = f'C{colored_nodes}'
      colored_nodes += 1
      node_colors[int(network.senders[i])] = c
      node_colors[int(network.receivers[i])] = c

  # Plot network.
  pos = nx.multipartite_layout(
      nx_graph, subset_key='time', align='horizontal', scale=-10
  )
  nx.draw(
      nx_graph,
      pos=pos,
      labels={node: int(network.nodes[node][0]) for node in nx_graph.nodes},
      font_color='k',
      node_color=[
          node_colors.get(node, 'lightgrey') for node in nx_graph.nodes
      ],
      edge_color=[
          edge_colors.get(edge, 'lightgrey' if solution is not None else 'grey')
          for edge in nx_graph.edges
      ],
      node_size=500,
      ax=ax,
      edgecolors=[node_edgecolors.get(node, 'k') for node in nx_graph.nodes],
  )

