from firesim import make_fuel_map, ignite_center, run
from threading import Thread
import numpy as np

import argparse
import datetime
import os


if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('--no-png-output', action='store_true')
    parser.add_argument('--n-cells', type=int)
    args = parser.parse_args()

    if args.n_cells:
        N_CELLS = args.n_cells
    else:
        N_CELLS = 200
    
    #create fuel map
    fuel = make_fuel_map(N_CELLS, tree_density=0.55, seed=42)

    # create array holding the states of the simulation
    state = np.zeros_like(fuel) # should the type be dtype=np.bool?
    ignite_center(state)

    sim_args = {
        'state_array': state,
        'fuel_array': fuel,
        'burn_rate': 3,
        'n_steps': 100,
        'ignition_prob': 0.2,
        'n_epochs': 10
    }

    # we will output images here
    dir_name =  f"sim_output_cells={str(state.shape)}_steps={str(sim_args['n_epochs']*sim_args['n_steps'])}_ignition_prob={str(sim_args['ignition_prob'])}_burn_rate={str(sim_args['burn_rate'])}_time={str(datetime.datetime.now())}"
    
    if not args.no_png_output:
        sim_args['save_dir'] = dir_name

        if not os.path.isdir(dir_name):
            os.mkdir(dir_name)

    else: sim_args['save_dir'] = None

    p = Thread(target=run, kwargs=sim_args)
    p.start()
    p.join()
    