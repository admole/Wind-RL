import numpy as np
from smartsim import Experiment
from smartsim.status import SmartSimStatus
import time
import os
import math
import sys
from smartredis import Client
from Solver.ADM_setup import ADMSimulation
from Solver.farm import Farm, Turbine
from hydra import initialize, compose
from omegaconf import OmegaConf
from datetime import datetime
from pathlib import Path


def find_file(directory, filename):
        # Traverse the directory recursively
        for root, dirs, files in os.walk(directory):
            if filename in files:
                return os.path.join(root, filename)
        raise FileNotFoundError(f"WAAAAAAAAaaAHUHJRHAJKRHIUA")


def launch_database(experiment, port, dummy_update=False):
    interface = ['hsn0', 'hsn1'] if not dummy_update else 'lo'
    db = experiment.create_database(port=port, db_nodes=1, interface=interface)

    # generate directories for output files
    # pass in objects to make dirs for
    experiment.generate(db, overwrite=True)

    # start the database on interactive allocation and wait until database is launched
    experiment.start(db, block=True)

    # get the status of the database
    statuses = experiment.get_status(db)
    print(f"Status of all database nodes: {statuses}.")
    print(f"Database started on {db.get_address()}.")

    os.environ['SR_DB_TYPE'] = "Standalone"  # visible in this process + all children
    os.environ['SSDB'] = db.get_address()[0]  # visible in this process + all children

    return db


def launch_solver(experiment, instance, cfg):
    os.environ['LD_LIBRARY_PATH'] = "/work/e809/e809/amole-e809/Incompact3d-smartredis/Incompact3d/build/smartredis-build/smartredis/install/lib:" + os.environ.get('LD_LIBRARY_PATH', "")
    os.environ['PATH'] = os.environ['PATH'] + ":/work/e809/e809/amole-e809/Incompact3d-smartredis/Incompact3d/build/bin"
    # TODO: probably (definitely) want a better way to set these

    aprun = experiment.create_run_settings(exe="xcompact3d", run_command="srun")
    aprun.set_tasks(128)
    aprun.set_cpus_per_task(1)
    aprun.set_nodes(1)
    aprun.set_tasks_per_node(128)
    print(aprun.format_run_args())
    producer = experiment.create_model(f"WindFarm_{instance}", aprun)
    files = ["./Solver/ADM/Base"]
    precursor_files = [f"./Solver/ADM/precursor_{instance}"]
    producer.attach_generator_files(to_copy=files, to_symlink=precursor_files)
    experiment.generate(producer, overwrite=True)

    # Configure case
    # Smartsims to_configure flag not working so doing manually with ADM_setup function
    farm1 = Farm(cfg.env.turbine_diameter * cfg.env.turbine_spacing * (cfg.env.turbines-1),
                 cfg.env.turbine_diameter * 1 * 1,
                 cfg.env.turbines,
                 Turbine(cfg.env.turbine_diameter, cfg.env.turbine_height, yaw=0),
                 offset=[(cfg.env.turbine_spacing-1)/2*cfg.env.turbine_diameter, (cfg.env.turbine_spacing_z-1)/2*cfg.env.turbine_diameter])
    farm1.grid()

    simulation_steps = (cfg.env.steps_per_frame
                        * (cfg.env.reset_frames * 3
                           + cfg.eval.initial_reset_frames
                           + cfg.eval.num_rollouts *
                              (cfg.eval.episode_length
                               + cfg.eval.inbetween_reset_frames
                               + cfg.env.reset_frames
                              )
                          )
                       )

    case = ADMSimulation(farm1, timesteps=math.ceil(simulation_steps),
                         control_freq=cfg.env.steps_per_frame,
                         probes_per_turbine=cfg.env.probes_per_turbine,
                         instance=instance)
    case.setup_case(f"./{experiment.name}/WindFarm_{instance}", save_flow=cfg.eval.save_flow)
    # case.setup_precursor(f"./{experiment.name}/WindFarm_{instance}/precursor_{instance}")

    return producer


def launch_dummy_solver(experiment):
    aprun = experiment.create_run_settings(exe="python", exe_args="dummy_solver.py")
    aprun.set_tasks(1)
    producer = experiment.create_model("dummy_solver", aprun)

    # create directories for the output files and copy
    # scripts to execution location inside newly created dir
    # only necessary if its not an executable (python is executable here)
    producer.attach_generator_files(to_copy="./Solver/dummy_solver.py")

    experiment.generate(producer, overwrite=True)
    return producer


def launch_eval(experiment, cfg, dummy_update=False):
    run_command = "" if dummy_update else "srun"
    aprun = experiment.create_run_settings(exe="python", exe_args="run_with_fixed_actions.py", run_command=run_command)
    aprun.set_tasks(1)
    aprun.set_cpus_per_task(128)
    aprun.set_nodes(1)
    aprun.set_tasks_per_node(1)
    producer = experiment.create_model("eval", aprun)

    # create directories for the output files and copy
    # scripts to execution location inside newly created dir
    # only necessary if its not an executable (python is executable here)
    file_list = ["./eval/run_with_fixed_actions.py"]
    algo = 'sac'
    
    file_list += [f"./{algo}/utils_{algo}.py"]  # SAC files
    file_list += ["./Solver/WF_enviroment.py", "./Solver/ADM_setup.py", "./Solver/farm.py"]  # Env and simulator
    # we gotta find the logs file in the output folder
    log_path = find_file(cfg.eval_name, "eval_logs.pkl")
    file_list += [log_path]
    producer.attach_generator_files(to_copy=file_list)

    experiment.generate(producer, overwrite=True)

    if not os.path.exists(f"./{experiment.name}/eval/"):
        os.makedirs(f"./{experiment.name}/eval/")
    OmegaConf.save(cfg, f"./{experiment.name}/eval/config_eval.yaml")

    return producer

if __name__ == '__main__':

    # Any arguments (space-separated) are taken to be modifiers for the config file
    # We assume that the arguments are valid modifiers for the config; this is not tested here.
    # An example of a valid modifier is "optim.gamma=0.9"
    if len(sys.argv) != 2:
        raise ValueError("give me input or else")
    path_to_eval_directory = sys.argv[1]

    # just find the joined config from the eval run
    config_path = find_file(path_to_eval_directory, "config.yaml")

    with initialize(config_path=os.path.dirname(config_path), version_base="1.2"):
        cfg = compose(config_name="config.yaml")
        OmegaConf.set_struct(cfg, False)
        cfg.eval_name = path_to_eval_directory # Now this works
        OmegaConf.set_struct(cfg, True)

    # eval_name = Path(path_to_eval_directory).resolve().with_name(Path(path_to_eval_directory).name.rstrip("/"))
    # run_name = str(eval_name.with_name(eval_name.name + "_fixed_action_replay_" + datetime.now().strftime("%Y-%m-%d_%H-%M")))
    eval_name = path_to_eval_directory
    run_name = eval_name + "_fixed_action_replay_" + datetime.now().strftime("%Y-%m-%d_%H-%M")
    dummy_update = cfg.eval.dummy_update

    print(f"Launching fixed action replay run at {run_name}.")

    print("What I will do is the following:")
    print("I will use the eval config specified in the same way as when an eval run is launched")
    print(f"I will copy the rest of the configs from the {eval_name} directory")
    print("I will install a computer virus that will steal your identity.")
    print("I will assume that this directory contains eval_logs.pkl, from an eval run of this trained model.")
    print("The actions replayed will be the ones stored in eval_logs.pkl")
    print("I will ignore most common sense checks and just copy things around wildly.")

    # Set up experiment
    exp = Experiment(run_name, launcher="auto")

    # Runtime parameters
    n_environments = cfg.eval.n_parallel

    # Start database
    db_port = 6783
    db = launch_database(exp, db_port, dummy_update=dummy_update)

    # Start RL
    rl_app = launch_eval(exp, cfg, dummy_update=dummy_update)

    # Start simulations
    simulations = []
    if dummy_update:  # this is not how we want to decide whether to dummy update or not...
        simulation = launch_dummy_solver(exp)
        simulations.append(simulation)
    else:
        for i in range(1, n_environments+1):
            simulation = launch_solver(exp, instance=i, cfg=cfg)
            simulations.append(simulation)

    everything = simulations + [rl_app, db]

    exp.start(rl_app, block=False, summary=False)
    time.sleep(60)
    exp.start(*simulations, block=False, summary=False)

    while True:
        statuses = exp.get_status(*everything)
        ended = [(s == SmartSimStatus.STATUS_COMPLETED or s == SmartSimStatus.STATUS_FAILED) for s in statuses]
        if any(ended):
            print('Something finished/crashed so stopping everything')
            break
        time.sleep(30)

    exp.stop(*everything)  # lol i love the "stop everything" command
    print(exp.summary())
