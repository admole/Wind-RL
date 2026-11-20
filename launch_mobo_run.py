import numpy as np
from smartsim import Experiment
from smartsim.status import SmartSimStatus
import time
import os
import math
from smartredis import Client
from Solver.ADM_setup import ADMSimulation
from Solver.farm import Farm, Turbine
from hydra import initialize, compose
import sys


def launch_database(experiment, port):
    db = experiment.create_database(port=port, db_nodes=1, interface=['hsn0', 'hsn1'])

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
    os.environ['LD_LIBRARY_PATH'] = "/work/e01/e01/amole/Incompact3d-smartredis/Incompact3d/build/smartredis-build/smartredis/install/lib:" + os.environ.get('LD_LIBRARY_PATH', "")
    os.environ['PATH'] = os.environ['PATH'] + ":/work/e01/e01/amole/Incompact3d-smartredis/Incompact3d/build/bin"
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
                        * (cfg.env.reset_frames*3 + cfg.optim.episode_steps) * cfg.optim.train_steps
                        +cfg.env.steps_per_frame*cfg.env.initial_reset_frames)  # This is horrible | Max: I agree, it truly is. horrendous
    case = ADMSimulation(farm1, timesteps=math.ceil(simulation_steps),
                         control_freq=cfg.env.steps_per_frame,
                         probes_per_turbine=cfg.env.probes_per_turbine,
                         instance=instance)
    case.setup_case(f"./{experiment.name}/WindFarm_{instance}")
    # case.setup_precursor(f"./{experiment.name}/WindFarm_{instance}/precursor_{instance}")

    return producer


def launch_bo(experiment, cfg, config_modifiers):
    aprun = experiment.create_run_settings(exe="python", exe_args="mobo.py " + " ".join(config_modifiers), run_command="srun")
    aprun.set_tasks(1)
    aprun.set_cpus_per_task(128)
    aprun.set_nodes(1)
    aprun.set_tasks_per_node(1)
    producer = experiment.create_model("mobo", aprun)

    # create directories for the output files and copy
    # scripts to execution location inside newly created dir
    # only necessary if its not an executable (python is executable here) 
    file_list = ["./mobo/mobo.py", "./mobo/utils_mobo.py", "./mobo/config_mobo.yaml"]  # mobo files
    file_list += ["./Solver/WF_enviroment.py", "./Solver/ADM_setup.py", "./Solver/farm.py"]  # Env and simulator files
    producer.attach_generator_files(to_copy=file_list)

    experiment.generate(producer, overwrite=True)
    return producer


if __name__ == '__main__':

    # The first command line argument determines the name of the run directory that everything is saved to
    run_name = sys.argv[1] if len(sys.argv) > 1 else "training_mobo"

    # Any subsequent arguments (space-separated) are taken to be modifiers for the config file
    # We assume that the arguments are valid modifiers for the config; this is not tested here.
    # An example of a valid modifier is "optim.gamma=0.9"
    config_modifiers = list(sys.argv[2:]) if len(sys.argv) > 1 else [""]

    # Read bo config
    initialize(config_path="./mobo/", version_base="1.2")
    cfg = compose(config_name="config_mobo.yaml", overrides=config_modifiers)

    # Set up experiment
    exp = Experiment(run_name, launcher="auto")

    # Runtime parameters
    n_environments = cfg.env.n_parallel

    # Start database
    db_port = 6783
    db = launch_database(exp, db_port)

    # Start RL
    rl_app = launch_bo(exp, cfg, config_modifiers)

    # Start simulations
    simulations = []
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
