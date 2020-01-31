import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import skimage.io
import numpy as np
import os
import cv2
from multiprocessing import Process

np.random.seed(42) # reproducibility

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

        red = cv2.resize(states_history[i], img_shape, interpolation=cv2.INTER_NEAREST)
        if color:
            green = cv2.resize(fuel_history[i], img_shape, interpolation=cv2.INTER_NEAREST)
            blue = np.zeros_like(green)
            im = np.stack([red, green, blue], axis=-1)
        else:
            im = red
        
        skimage.io.imsave(f'{dir_name}/frame{padded_n}.png', im)
    


kernel3by3 = np.array([[1,1,1],[1,0,1],[1,1,1]]) # 3x3
kernel5by5 = 15 * gkern()
kernel5by5[2,2] = 0
kernel5by5 = kernel3by3


def run(n_cells=1000, tree_density=0.525123333333333333333, burn_rate=3, n_steps=10, ignition_prob=0.2, n_epochs=10, save=False, color=False):
    # initialization n_cells x n_cells could correspond to 1000x1000m

    if save:
        # we will output images here
        dir_name =  f'cells={str(n_cells)}steps={str(n_epochs*n_steps)}ignition_prob={str(ignition_prob)}tree_density={str(tree_density)}, burn_rate={str(burn_rate)}'
        if not os.path.isdir(dir_name):
            os.mkdir(dir_name)

    # fuel corresponds to the ammount of fuel in each cell (0-255)
    fuel = np.zeros((n_cells, n_cells), dtype=np.uint8)

    # fill tree_density percentage of cells with trees
    # (e.g. 0.55 corresponds to 55% cells having trees (fuel) and 45% being empty)
    n_trees = int(n_cells**2 * tree_density) 
    trees = np.random.randint(0 ,n_cells, (n_trees,2))
    fuel[trees[:,0], trees[:,1]] = np.random.randint(0,255,n_trees)


    state = np.zeros_like(fuel) # should the type be dtype=np.bool?
    ignite_center(state)
    states_history = np.array(np.zeros((n_steps,*state.shape)), dtype=np.uint8)
    ignitions_history = np.zeros((n_steps,*state.shape), dtype=np.uint8)
    fuel_history = np.zeros((n_steps,*state.shape), dtype=np.uint8)


    for e in range(n_epochs):
        print(f'epoch:{e}')
        # run simulation
        for i in range(n_steps):


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

        if save:
            save_images(states_history, fuel_history, dir_name, start_number=e)

if __name__ == '__main__':

    #run(n_cells=1000, tree_density=0.525, burn_rate=3, n_steps=10, ignition_prob=0.2, n_epochs=10, save=False, color=False)
    N_CELLS = 1000
    N_DRONES = 10
    

    sim_arguments = {
        'n_cells': N_CELLS,
        'tree_density': 0.525,
        'burn_rate': 3,
        'n_steps': 10,
        'ignition_prob': 0.2,
        'n_epochs': 10,
        'save': True,
        'color': False
    }


    p = Process(target=run, kwargs=sim_arguments)
    p.start()
    p.join()

    drones = np.random.randint(N_DRONES, 2)
    

