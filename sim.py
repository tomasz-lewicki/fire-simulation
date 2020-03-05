from multiprocessing import Process
import os
import datetime

import numpy as np
import scipy.signal
import cv2

import matplotlib.pyplot as plt
import skimage.io

import time

def ignite_center(state):
    center_x = int(state.shape[0]/2)
    center_y = int(state.shape[1]/2)
    state[center_x-3:center_x+3,center_y-3:center_y+3] = 1
    

def gkern(l=5, sig=1.):
    """
    creates gaussian kernel with side length l and a sigma of sig
    """

    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    xx, yy = np.meshgrid(ax, ax)

    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sig))

    return kernel / np.sum(kernel)

def save_images(states_history, fuel_history, dir_name, start_number, img_shape= (2000,2000), color=True): #TODO: have corresponding dict {state='blue', fuel='red'}
    n_images = len(states_history)
    for i in range(n_images):

        padded_n = '{0:05d}'.format(start_number*n_images+i)
        
        red = states_history[i]
        red[red>0] = 255 # make fire cells max intensity
        green = fuel_history[i]
        green[red>0] = 0

        red = cv2.resize(red, img_shape, interpolation=cv2.INTER_NEAREST)
        if color:
            green = cv2.resize(green, img_shape, interpolation=cv2.INTER_NEAREST)
            blue = np.zeros_like(green)
            im = np.stack([red, green, blue], axis=-1)
        else:
            im = red
        
        skimage.io.imsave(f'{dir_name}/frame{padded_n}.png', im)
    


kernel3by3 = np.array([[1,1,1],[1,0,1],[1,1,1]]) # 3x3
kernel5by5 = 15 * gkern()
kernel5by5[2,2] = 0
kernel5by5 = kernel3by3


def run(state_array, fuel_array, burn_rate=3, n_steps=10, ignition_prob=0.2, n_epochs=10, save_dir=None, loop_min_dur=1):

    state = state_array
    fuel = fuel_array

    states_history = np.array(np.zeros((n_steps,*state.shape)), dtype=np.uint8)
    ignitions_history = np.zeros((n_steps,*state.shape), dtype=np.uint8)
    fuel_history = np.zeros((n_steps,*state.shape), dtype=np.uint8)

    for e in range(n_epochs):
        print(f'epoch:{e}')
        # run simulation
        for i in range(n_steps):

            iter_start = time.monotonic()
            # calculate new ignitons
            count_neighbors_on_fire = scipy.signal.convolve2d(state, kernel5by5, 'same')
            ignitions = (count_neighbors_on_fire * np.random.random(state.shape) > ignition_prob) * fuel
            state += ignitions
            

            #update fuel status
            on_fire_mask = state > 0
            burned_out_mask = fuel < burn_rate
            fuel[on_fire_mask] -= burn_rate 
            fuel[burned_out_mask] = 0
            state[burned_out_mask] = 0

            # update histories
            ignitions_history[i] = ignitions
            states_history[i] = state
            fuel_history[i] = fuel
            iter_stop = time.monotonic()

            iter_delay = iter_stop - iter_start
            if(iter_delay<loop_min_dur):
                time.sleep(loop_min_dur-iter_delay)

        if save_dir:
            save_images(states_history, fuel_history, save_dir, start_number=e)


def make_fuel_map(n_cells, tree_density, seed=42):
    # fuel corresponds to the ammount of fuel in each cell (0-255)

    fuel = np.zeros((n_cells, n_cells), dtype=np.uint8)
    
    # fill tree_density percentage of cells with trees
    # (e.g. 0.55 corresponds to 55% cells having trees (fuel) and 45% being empty)
    n_trees = int(n_cells**2 * tree_density) 
    trees = np.random.randint(0 ,n_cells, (n_trees,2))
    
    np.random.seed(seed) # reproducibility
    fuel[trees[:,0], trees[:,1]] = np.random.randint(0,255,n_trees)

    return fuel


if __name__ == '__main__':

    N_CELLS = 100
    N_DRONES = 10
    
    #create fuel map
    fuel = make_fuel_map(N_CELLS, tree_density=0.55, seed=42)

    # create array holding the states of the simulation
    state = np.zeros_like(fuel) # should the type be dtype=np.bool?
    ignite_center(state)

    sim_args = {
        'state_array': state,
        'fuel_array': fuel,
        'burn_rate': 3,
        'n_steps': 10,
        'ignition_prob': 0.2,
        'n_epochs': 10
    }

    # we will output images here
    dir_name =  f"sim_output_cells={str(state.shape)}_steps={str(sim_args['n_epochs']*sim_args['n_steps'])}_ignition_prob={str(sim_args['ignition_prob'])}_burn_rate={str(sim_args['burn_rate'])}_time={str(datetime.datetime.now())}"
    
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)

    sim_args['save_dir'] = dir_name

    p = Process(target=run, kwargs=sim_args)
    p.start()
    p.join()

    # drones = np.random.randint((N_DRONES, 2))
    

