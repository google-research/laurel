from pathlib import Path

import jax
import numpy as np
from cluster.settings import read_params_from_cmdline, save_metrics_params
from flax.training import checkpoints
from orbax.checkpoint import PyTreeCheckpointer
from tqdm import tqdm

from laurel import control, mdp

# Read settings from yaml file.
params = read_params_from_cmdline()
params = params._mutable_copy()
params.update(dict(params.conf))
dir_ = Path(params.working_dir)

# Seeding
seed = params.seed * 1000
rng = np.random.default_rng(seed)
key = jax.random.PRNGKey(42)

# Initialize training MDP.
env = mdp.MiddleMileMDP(
    num_hubs=10,
    timesteps=50,
    num_trucks_per_step=10,
    max_truck_duration=5,
    num_parcels=200,
    mean_route_length=10,
    cut_capacities=0,
    unit_weights=True,
    unit_capacities=True,
)

if params.function_type == 'linear':
    controller = control.LinearPPO(
        rng=rng,
        key=key,
        env=env,
        num_rollouts_total=params.num_rollouts_total,
        num_rollouts=params.num_rollouts,
        num_actor_updates=params.num_actor_updates,
        num_critic_updates=params.num_critic_updates,
        actor_lr=params.actor_lr,
        critic_lr=params.critic_lr,
        exploration_inv_temperature=params.exploration_inv_temperature,
        clip_eps=params.clip_eps,
        kl_stop=params.kl_stop,
        batch_size=params.batch_size,
    )
else:
    controller = control.GNN_PPO(
        rng=rng,
        key=key,
        env=env,
        num_rollouts_total=params.num_rollouts_total,
        num_rollouts=params.num_rollouts,
        num_actor_updates=params.num_actor_updates,
        num_critic_updates=params.num_critic_updates,
        actor_lr=params.actor_lr,
        critic_lr=params.critic_lr,
        num_feature_graph_steps=params.num_feature_graph_steps,
        exploration_inv_temperature=params.exploration_inv_temperature,
        clip_eps=params.clip_eps,
        kl_stop=params.kl_stop,
        batch_size=params.batch_size,
    )
losses_actor, losses_critic, exploration, performance = controller.train(
    pb_epoch=tqdm
)

# Save results.
np.save(dir_ / 'losses_actor.npy', losses_actor)
np.save(dir_ / 'losses_critic.npy', losses_critic)
np.save(dir_ / 'exploration.npy', exploration)
np.save(dir_ / 'performance.npy', performance)

# Save trained parameters.
# Orbax disabled because of locking error.
# PyTreeCheckpointer().save(dir_ / 'actor_params', controller._actor_params)
# PyTreeCheckpointer().save(dir_ / 'critic_params', controller._critic_params)
checkpoints.save_checkpoint(dir_ / 'actor_params', controller._actor_params, 0)
checkpoints.save_checkpoint(
    dir_ / 'critic_params', controller._critic_params, 0
)


# Announce results to cluster-utils.
save_metrics_params({'final_mean_perf': performance[-1].mean()}, params)
