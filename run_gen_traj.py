from typing import Dict, Optional, Tuple

import subprocess
from subprocess import Popen
import configparser
import ast
import random
import time
from time import gmtime, strftime
import sys
import itertools
# import inquirer
import tempfile
import socket
import os
import pdb

#Popen('mkdir -p save/cmds/', shell=True).wait()
#Popen('mkdir -p save/logs/', shell=True).wait()
#Popen('mkdir -p save/logs/stdout/', shell=True).wait()

jobid = 0
def runcmd(cmd: str,
           identifier: str = 'train',
           queue_type: str = 'short',
           time_limit: Optional[int] = None,
           exclude_nodes: str = '',
           stdout_to_log: bool = True,
           wait: bool = False):
    global jobid
    assert queue_type in ['noslurm', 'debug', 'short', 'long']
    log_fname = 'save/logs/stdout/job_{}_{}_{}.log'.format(
        identifier, int(time.time()), jobid)
    jobid += 1
    # write SLURM job id then run the command
    write_slurm_id = True
    if write_slurm_id:
        script_file = tempfile.NamedTemporaryFile(mode='w', delete=False,
                             dir='./save/cmds/', prefix='.', suffix='.slurm.sh')
        script_file.write('echo "slurm job id: $SLURM_JOB_ID"\n')
        script_file.write('echo ' + cmd + '\n')
        script_file.write('echo "host: $HOSTNAME"\n')
        script_file.write('echo "cuda: $CUDA_VISIBLE_DEVICES"\n')
        #script_file.write('nvidia-smi -i $CUDA_VISIBLE_DEVICES\n')
        script_file.write(cmd)
        script_file.close()
        # use this to restrict runs to current host
        #hostname = socket.gethostname()
        # cmd = ' -w {} bash '.format(hostname) + script_file.name
        cmd = 'bash ' + script_file.name

    # time_limit in hours
    if type(time_limit) == int and time_limit > 0:
        time_str = "-t {:02d}:00:00".format(time_limit)
    else:
        time_str = ""

    if exclude_nodes:
        exclude_str = "--exclude {}".format(exclude_nodes)
    else:
        exclude_str = ""

    #print(cmd)
    if queue_type == 'noslurm':
        print("WARNING: Running job without SLURM, use with caution.")
        proc = Popen(cmd, shell=True).wait()

    # Run a slurm job
    elif not stdout_to_log:
        proc = Popen('srun -p {} '.format(queue_type) + time_str + \
                ' {} --gres gpu:1 --pty '.format(exclude_str) + cmd, shell=True)
    else:
        # Use logging without stdout to terminal
        proc = Popen('srun -p {} '.format(queue_type) + time_str + \
                ' {} --gres gpu:1 -o {} --open-mode=append '.format(exclude_str, log_fname) + cmd, shell=True)

    if wait and queue_type != 'noslurm':
        proc.wait()
    # Run the command without slurm (or within an interactive slurm session)
    # Popen(cmd, shell=True).wait()
    return proc

def product_dict(**kwargs) -> Tuple[int, Dict]:
    "Cartesian product for parameter grid search"
    keys = kwargs.keys()
    vals = kwargs.values()
    for id, instance in enumerate(itertools.product(*vals)):
        yield id, dict(zip(keys, instance))

def parse_dict_literals(config_as_dict: Dict[str, str]) -> Dict:
    '''
    'configparser' stores every python literal
    as a string. This function parses these
    strings back to python literals.
    '''
    return {key:ast.literal_eval(val) \
        for key,val in config_as_dict.items()}

def parse_config_section(
    default_file: str, cfg_file: str, section_name: str) -> Dict:
    config = configparser.ConfigParser()

    config.read(default_file)
    params = parse_dict_literals(dict(config[section_name]))

    config.read(cfg_file)
    params.update(parse_dict_literals(dict(config[section_name])))
    return params

def generate_save_dir_name(prefix: str) -> str:
    timeStamp = strftime('%d-%b-%Y-%H-%M-%S', gmtime())
    return prefix + '_' + timeStamp \
        + '_{:0>6d}'.format(random.randint(0, 10e6))


default_cfg_file = 'configs/gen_traj/default.ini'
cfg_file = sys.argv[1]
print("Loading configs from: {}".format(cfg_file))

args = parse_config_section(
    default_cfg_file, cfg_file, 'JOB_PARAMS')

script_args = parse_config_section(
    default_cfg_file, cfg_file, 'SCRIPT_PARAMS')

RUN_FILE = script_args['run_file']
WAIT = script_args['wait']
REDIRECT_STDOUT = script_args['redirect_stdout']
QUEUE_TYPE = script_args['slurm_queue_type']
#VISDOM_ENV_NAME = script_args['visdom_env_name']
RUN_ID = script_args['run_id']
DRY_RUN = script_args['dry_run']
COMMON_SAVE_SUBDIR = script_args['common_save_subdir']
CUDA_LAUNCH_BLOCKING = script_args['cuda_launch_blocking']
EXCLUDE_NODES = script_args['exclude_nodes']
TIME_LIMIT = script_args['time_limit']
if TIME_LIMIT == 0:
    TIME_LIMIT = None
print("Constants loaded.")

# Hardcoded arguments #TODO: Move these to config file
#k_hot_embed_size = 32
#one_hot_embed_size = 100

if COMMON_SAVE_SUBDIR:
    save_subdir_prefix = generate_save_dir_name(RUN_ID)

if __name__ == '__main__':
    LAUNCH_COUNT = 0
    for local_job_id, params in product_dict(**args):
        params['id'] = RUN_ID
        env_name = ("{id}_"
                    "--base-dir-{base_dir}_"
                    "--n-demos-{n_demos}_"
                    "--ckpt-{ckpt}").format(**params)

        env_name = env_name.replace('/', '-')
        env_name = env_name.replace(' ', '-')

        params['id'] = env_name 

        # Special fix for buffer size
        params['RUN_FILE'] = RUN_FILE
        cmd = "python {RUN_FILE} \
                --id {id} \
                --base_dir {base_dir} \
                --ckpt {ckpt} \
                --n_demos {n_demos}".format(**params)

        if CUDA_LAUNCH_BLOCKING:
            cmd = "CUDA_LAUNCH_BLOCKING=1 " + cmd
        if not DRY_RUN:
            proc = runcmd(cmd,
                identifier=env_name,
                time_limit=TIME_LIMIT,
                queue_type=QUEUE_TYPE,
                exclude_nodes=EXCLUDE_NODES,
                stdout_to_log=REDIRECT_STDOUT,
                wait=WAIT)
            print("Launced")
        else:
            print("Using dry_run == True, jobs not launched!")
        print(params)
        LAUNCH_COUNT += 1

    if QUEUE_TYPE != 'noslurm':
        print("{} job(s) launched!".format(LAUNCH_COUNT))

    if DRY_RUN:
        print("Using dry_run == True, jobs not launched!")