from pathlib import Path

from cluster.settings import read_params_from_cmdline, save_metrics_params
import numpy as np
from orbax.checkpoint import PyTreeCheckpointer
import jax
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

controller = control.LinearPPO(
    rng,
    key,
    env,
    params.num_epochs,
    params.num_rollouts,
    params.num_actor_updates,
    params.num_critic_updates,
    params.actor_lr,
    params.critic_lr
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
from flax.training import checkpoints
checkpoints.save_checkpoint(dir_ / 'actor_params', controller._actor_params, 0)
checkpoints.save_checkpoint(
    dir_ / 'critic_params', controller._critic_params, 0
)


# Announce results to cluster-utils.
save_metrics_params({'final_mean_perf': performance[-1].mean()}, params)
