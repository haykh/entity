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


fig = plt.figure(dpi=300, figsize=(8,8), facecolor='white')

ax1= fig.add_axes([0.05,0.5,0.83,0.4])
ax1.plot(time_new,em_new, color='blue', label=r'$e^-$, new')
ax1.plot(time_new,ep_new, color='red', label=r'$e^+$, new')
ax1.legend()
ax1.set_ylim(0,1.8e5)
ax1.set_xlim(0,100)
    
fig.savefig("nparts.png",dpi=300,bbox_inches='tight')
plt.close()
