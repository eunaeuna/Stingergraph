import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

fname = 'delta_count.txt'
data = np.loadtxt(fname, dtype='int')
x = np.arange(0, 20)

for i in range(0,20):
    y = i * 10
    plt.plot(x,data[y,:])


fig, axes = plt.subplots(4,5, figsize=(15,6), facecolor='w', edgecolor='k')
"""
for i in range(0,4):
   for j in range(0,5):
       for k in range(0,10):
           #pos = i * j + k
           pos = 50 * i + 10 * j + k
           axes[i,j].plot(x,data[pos,:])
           axes[i,j].set_title(str((i*5)+j+1)+"%")
"""
for i in range(0,4):
   for j in range(0,5):
       axes[i,j].set_title(str((i*5)+j+1)+"%")
       pos = 50 * i + 10 * j
       axes[i,j].plot(x,data[pos,:],color='#D9E3FC')
       pos = pos + 1
       axes[i,j].plot(x,data[pos,:],color='#C5D6FC')
       pos = pos + 1
       axes[i,j].plot(x,data[pos,:],color='#ADC4F9')
       pos = pos + 1
       axes[i,j].plot(x,data[pos,:],color='#91AFF5')
       pos = pos + 1
       axes[i,j].plot(x,data[pos,:],color='#789CF0')
       pos = pos + 1
       axes[i,j].plot(x,data[pos,:],color='#5E89EC')
       pos = pos + 1
       axes[i,j].plot(x,data[pos,:],color='#3E6EDC')
       pos = pos + 1
       axes[i,j].plot(x,data[pos,:],color='#235CE1')
       pos = pos + 1
       axes[i,j].plot(x,data[pos,:],color='#114DD8')
       pos = pos + 1
       axes[i,j].plot(x,data[pos,:],color='#1E07F0')

#red_patch = mpatches.Patch(color='red', label='The red data')
#plt.legend(handles=[red_patch])
plt.show()

