#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import os

# to convert the output png files:
# ffmpeg -i out%03d.png -c:v libx264 -crf 0 -preset veryslow -c:a libmp3lame -b:a 320k output.mp4

OUT_DIR_NAME = 'drones_positions_plt'
N_DRONES = 50
N_CELLS = 1000
VELOCITY = 10

if not os.path.isdir(OUT_DIR_NAME):
    os.mkdir(OUT_DIR_NAME)

# init location matrix
drones_xy = np.random.random((N_DRONES,2)) * N_CELLS

# generate heading for all drones
drones_heading = np.random.random((N_DRONES,1)) * 360

# init velocity matrix
drones_velocity_xy = VELOCITY * np.hstack([np.sin(drones_heading), np.cos(drones_heading)])


# sin_cos_heading = np.hstack([np.sin(drones_heading), np.cos(drones_heading)])
# np.sum(np.power(sin_cos_heading,2), axis=1) # sanity check, should be all ones

for step in range(1000):
    drones_xy += drones_velocity_xy
    
    for i, ((x,y),(vx,vy)) in enumerate(zip(drones_xy, drones_velocity_xy)):
        if x<0 or x>N_CELLS:
            drones_velocity_xy[i,0] = -vx
        if y<0 or y>N_CELLS:
            drones_velocity_xy[i,1] = -vy
    
    # plot stuff
    if step % 1 == 0:
        plt.figure()
        plt.xlim(0, N_CELLS)
        plt.ylim(0, N_CELLS)
        plt.scatter(drones_xy[:,0], drones_xy[:,1])
        ax = plt.axes()
        
        # plot vectors
        for (x,y),(vx,vy) in zip(drones_xy, drones_velocity_xy):
            ax.arrow(x, y, 3*vx, 3*vy, head_width=0.05, head_length=10, fc='k', ec='k')
        
        plt.savefig(f'{OUT_DIR_NAME}/out{step:03d}.png', dpi=300)

