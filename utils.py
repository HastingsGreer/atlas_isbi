
import footsteps

import matplotlib.pyplot as plt

def show(vol):
    plt.imshow(vol[0, 0, 50].cpu().detach())
    footsteps.plot()

def warplines(vol):
    plt.imshow(vol[0, 0, 0].cpu().detach() * 0)
    plt.contour(vol[0,1, 50].cpu().detach())
    plt.contour(vol[0,2, 50].cpu().detach())
    footsteps.plot()
