import os
import matplotlib.pyplot as plt

groupA = ['20-30-1.txt', '20-30-2.txt', '20-30-3.txt', '20-30-4.txt']
groupA_xs = []
groupA_ys = []
groupA_names = []


groupB = ['10-30-2.txt', '20-30-2.txt', '40-30-2.txt', '60-30-2.txt']
groupB_xs = []
groupB_ys = []
groupB_names = []


groupC = ['20-10-2.txt', '20-30-2.txt', '20-90-2.txt', '20-120-2.txt']
groupC_xs = []
groupC_ys = []
groupC_names = []


groupD = ['10-10-4.txt', '20-30-3.txt', '40-90-2.txt', '60-120-1.txt']
groupD_xs = []
groupD_ys = []
groupD_names = []

for file in os.listdir("./results/"):
    f = open("./results/" + file)
    d = eval(f.readlines()[0])
    
    if file in groupA:
        groupA_xs.append(list(d.keys()))
        groupA_ys.append(list(d.values()))
        groupA_names.append(file)
    if file in groupB:
        groupB_xs.append(list(d.keys()))
        groupB_ys.append(list(d.values()))
        groupB_names.append(file)

    if file in groupC:
        groupC_xs.append(list(d.keys()))
        groupC_ys.append(list(d.values()))
        groupC_names.append(file)

    if file in groupD:
        groupD_xs.append(list(d.keys()))
        groupD_ys.append(list(d.values()))
        groupD_names.append(file)


for xs, ys, name in zip(groupA_xs, groupA_ys, groupA_names):
    namelis = name.replace(".txt", "").split("-")
    n = "b:" + namelis[0] + "; " + r'$\alpha$:' + namelis[1] + "; k:" + namelis[2]
    plt.plot(xs, ys, label = n, marker='^', markersize=4)


#get handles and labels
handles, labels = plt.gca().get_legend_handles_labels()
print(handles, labels)
#specify order of items in legend
order = [0, 2, 1, 3]

#add legend to plot
plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order]) 


import numpy as np
plt.xlabel("Noise Level " + r'($\sigma$)')
plt.yticks(np.arange(0, 1.05, 0.05))
plt.ylabel("Test Accuracy")
plt.title("Varying Fingerprint Size")
plt.savefig('plt1.png', dpi=400)
plt.show()



for xs, ys, name in zip(groupB_xs, groupB_ys, groupB_names):
    namelis = name.replace(".txt", "").split("-")
    n = "bin:" + namelis[0] + "; " + r'$\alpha$:' + namelis[1] + "; k:" + namelis[2]
    plt.plot(xs, ys, label = n, marker='^', markersize=4)


#get handles and labels
handles, labels = plt.gca().get_legend_handles_labels()
print(handles, labels)
#specify order of items in legend
order = [0, 2, 3, 1]

#add legend to plot
plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order]) 
#plt.legend()
plt.xlabel("Noise Level " + r'($\sigma$)')
plt.yticks(np.arange(0, 1.05, 0.05))
plt.ylabel("Test Accuracy")
plt.title("Varying Bin Size")
plt.savefig('plt2.png', dpi=400)
plt.show()

for xs, ys, name in zip(groupC_xs, groupC_ys, groupC_names):
    namelis = name.replace(".txt", "").split("-")
    n = "bin:" + namelis[0] + "; " + r'$\alpha$:' + namelis[1] + "; k:" + namelis[2]
    plt.plot(xs, ys, label = n, marker='^', markersize=4)

#get handles and labels
handles, labels = plt.gca().get_legend_handles_labels()
print(handles, labels)
#specify order of items in legend
order = [0, 2, 1, 3]

#add legend to plot
#plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order]) 

plt.legend()
plt.xlabel("Noise Level " + r'($\sigma$)')
plt.yticks(np.arange(0, 1.05, 0.05))
plt.ylabel("Test Accuracy")
plt.title("Varying Intensity Threshold")
plt.savefig('plt3.png', dpi=400)

plt.show()



for xs, ys, name in zip(groupD_xs, groupD_ys, groupD_names):
    namelis = name.replace(".txt", "").split("-")
    n = "bin:" + namelis[0] + "; " + r'$\alpha$:' + namelis[1] + "; k:" + namelis[2]
    plt.plot(xs, ys, label = n, marker='^', markersize=4)

#get handles and labels
handles, labels = plt.gca().get_legend_handles_labels()
print(handles, labels)
#specify order of items in legend
order = [1, 2, 3, 0]


#add legend to plot
plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order]) 

plt.xlabel("Noise Level " + r'($\sigma$)')
plt.yticks(np.arange(0, 1.05, 0.05))
plt.ylabel("Test Accuracy")
plt.title("Low and High Complexity Parameterizations")
plt.savefig('plt4.png', dpi=400)

plt.show()

