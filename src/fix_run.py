"""Run with experiment name as parameters, job file for failed or unstarted jobs will be generated."""

import os
import sys
from glob import glob

import ruamel.yaml as yaml
from tqdm import tqdm

# Read in names of original job + previous fixes, generate new name.
name = sys.argv[1]
names = [name]
name_fix = f'{name}-fix'
i = 2
while os.path.exists(f'dat/runs/{name_fix}/'):
    names.append(name_fix)
    name_fix = f'{name}-fix{i}'
    i += 1

# Read in original job yaml. This will be the base for the new job.
with open(f'run/experiments/{name}.yaml') as f:
    conf = yaml.safe_load(f.read().replace('&name ' + name, '&name ' + name_fix))

# Generate list of grid search jobs
param0 = conf['hyperparam_list'][0]
experiments = [[[param0['param'], v]] for v in param0['values']]
experiments_ = []
for param in conf['hyperparam_list'][1:]:
    for experiment in experiments:
        for v in param['values']:
            experiments_.append(experiment + [[param['param'], v]])
    experiments = experiments_
    experiments_ = []

# Check which experiments finished successfully
for n in names:
    for g in tqdm(glob(f"dat/runs/{n}/working_directories/*/")):
        if not os.path.exists(f"{g}/metrics.csv"):
            continue
        with open(f'{g}/settings.json') as f:
            settings = yaml.safe_load(f)
            experiment = []
            for param in conf['hyperparam_list']:
                p = param['param']
                done = False
                if 'conf' in settings:
                    for n, v in settings['conf']:
                        if n == p:
                            experiment.append([n, v])
                            done = True
                if p in settings and not done:
                    experiment.append([p, settings[p]])
                    done = True
                if not done:
                    raise ValueError(f"Missing parameter '{p}' in {settings}")
            experiments.remove(experiment)

conf["hyperparam_list"] = [{'param': 'conf', 'values': experiments}]

if experiments:
    with open(f'run/experiments/{name_fix}.yaml', 'w') as f:
        yaml.dump(conf, f)
else:
    print("All experiments ran successfully; No job file generated.")
