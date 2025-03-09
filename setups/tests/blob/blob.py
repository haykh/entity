import h5py
import numpy as np
import matplotlib.pyplot as plt

f = open("report", "r")
Lines = f.readlines()
f.close()

em_new = []
ep_new = []
time_new = []
for i in range (len(Lines)):
    line = Lines[i]
    line = line.strip()
    arr = line.split()
    
    if (len(arr)>0 and arr[0]=='species'):
        nparts = arr[2].split("..")
        if (nparts[0]=="(e-_p)"):
            em_new.append(float(nparts[-1]))
        if (nparts[0]=="(e+_p)"):
            ep_new.append(float(nparts[-1]))

    if (len(arr)>0 and arr[0]=='Time:'):
        time_new.append(float(arr[1]))

f = h5py.File('blob.h5', 'r')

Nsteps = len(f.keys())
print(list(f['Step0'].keys()))

for i in range (Nsteps):
    print (i)
    fig = plt.figure(dpi=300, figsize=(8,8), facecolor='white')

    densMax = max(np.max(f['Step'+str(i)]['fN_1']),np.max(f['Step'+str(i)]['fN_2']))
    print(densMax)
    ax1 = fig.add_axes([0.05,0.05,0.4,0.4])
    im1=ax1.pcolormesh(f['Step'+str(i)]['X1'],f['Step'+str(i)]['X2'],f['Step'+str(i)]['fN_1'],cmap='turbo',vmin=0,vmax=1.0)
    ax1.set_title(r"$N_1$")
    ax1.vlines(0,-10.0,10.0,color='white')

    ax1 = fig.add_axes([0.48,0.05,0.4,0.4])
    ax1.pcolormesh(f['Step'+str(i)]['X1'],f['Step'+str(i)]['X2'],f['Step'+str(i)]['fN_2'],cmap='turbo',vmin=0,vmax=1.0)
    ax1.set_yticklabels([])
    ax1.set_title(r"$N_2$")
    ax1.vlines(0,-10.0,10.0,color='white')

    ax4cb = fig.add_axes([0.89, 0.05, 0.01, 0.4])
    cbar4 = fig.colorbar(im1,cax=ax4cb)

    ax1= fig.add_axes([0.05,0.5,0.83,0.4])
    ax1.plot(time_new,em_new, color='blue', label=r'$e^-$, new')
    ax1.plot(time_new,ep_new, color='red', label=r'$e^+$, new')
    ax1.legend()
    ax1.set_ylim(0,1.8e5)
    ax1.set_xlim(0,100)
    ax1.vlines(i, 0,1.8e5, color='green',linewidth=0.6)
    
    
    fig.savefig("%05d"%i+".png",dpi=300,bbox_inches='tight')
    plt.close()
