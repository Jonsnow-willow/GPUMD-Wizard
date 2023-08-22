import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (6, 6)
plt.rcParams.update({"font.size": 10, "text.usetex": False})

loss = np.loadtxt("loss.out")
energy = np.loadtxt("energy_train.out")
force = np.loadtxt("force_train.out")
virial_initial = np.loadtxt("virial_train.out")
virial= virial_initial[np.logical_not(np.any(virial_initial == -1e6, axis=1))]
loss = loss[:, 2:7]

fig, axes = plt.subplots(2, 2)

axes[0, 0].loglog(loss)
axes[0, 0].annotate("(a)", xy=(0.0, 1.1), xycoords="axes fraction", va="top", ha="right")
axes[0, 0].set_xlabel("Generation")
axes[0, 0].set_ylabel("Loss functions")
axes[0, 0].legend(["L1", "L2", "Energy", "Force", "Virial"], loc="lower left")
axes[0, 0].set_xlim([0, loss.shape[0]])

axes[0, 1].plot(energy[:, 1], energy[:, 0], ".", markersize=10, color=[0, 0.45, 0.74])
axes[0, 1].plot(axes[0, 1].get_xlim(), axes[0, 1].get_xlim(), linewidth=2, color=[0.85, 0.33, 0.1])
axes[0, 1].annotate("(b)", xy=(0.0, 1.1), xycoords="axes fraction", va="top", ha="right")
axes[0, 1].set_xlabel("DFT energy (eV/atom)")
axes[0, 1].set_ylabel("NEP energy (eV/atom)")
x_min = np.min(energy[:, 1]) * 1.1
x_max = np.max(energy[:, 1]) * 1.1
y_min = np.min(energy[:, 1]) * 1.1
y_max = np.max(energy[:, 1]) * 1.1

axes[0, 1].set_xlim(x_min, x_max)
axes[0, 1].set_ylim(y_min, y_max)

axes[1, 0].plot(force[:, 3:6], force[:, 0:3], ".", markersize=10, color=[0, 0.45, 0.74])
axes[1, 0].plot(axes[1, 0].get_xlim(), axes[1, 0].get_xlim(), linewidth=2, color=[0.85, 0.33, 0.1])
axes[1, 0].annotate("(c)", xy=(0.0, 1.1), xycoords="axes fraction", va="top", ha="right")
axes[1, 0].set_xlabel("DFT force (eV/Å)")
axes[1, 0].set_ylabel("NEP force (eV/Å)")
x_min = np.min(force[:, 3:6]) * 1.1
x_max = np.max(force[:, 3:6]) * 1.1
y_min = np.min(force[:, 3:6]) * 1.1
y_max = np.max(force[:, 3:6]) * 1.1

axes[1, 0].set_xlim(x_min, x_max)
axes[1, 0].set_ylim(y_min, y_max)

axes[1, 1].plot(virial[:, 0:6], virial[:, 6:12], ".", markersize=10, color=[0, 0.45, 0.74])
axes[1, 1].plot(axes[1, 1].get_xlim(), axes[1, 1].get_xlim(), linewidth=2, color=[0.85, 0.33, 0.1])
axes[1, 1].annotate("(d)", xy=(0.0, 1.1), xycoords="axes fraction", va="top", ha="right")
axes[1, 1].set_xlabel("DFT virial (eV/atom)")
axes[1, 1].set_ylabel("NEP virial (eV/atom)")
x_min = np.min(virial[:, 0:6]) * 1.1
x_max = np.max(virial[:, 0:6]) * 1.1
y_min = np.min(virial[:, 0:6]) * 1.1
y_max = np.max(virial[:, 0:6]) * 1.1

axes[1, 1].set_xlim(x_min, x_max)
axes[1, 1].set_ylim(y_min, y_max)

plt.subplots_adjust(hspace=0.4, wspace=0.3)

plt.savefig("Fig1.png")
plt.show()