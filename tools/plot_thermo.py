import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

filename1 = 'test/thermo.out'
filename2 = 'test/test.out'
column = 3
num = -1
with open(filename1, 'r') as f:
    lines = f.readlines()
    lines = [line.strip() for line in lines]
    lines = [line.split() for line in lines]
    lines = [line for line in lines if len(line) == 12]  
    lines = [[float(i) for i in line] for line in lines]
    data = np.array(lines)
thermo = data[:, column] / num
with open(filename2, 'r') as f:
    lines = f.readlines()
    lines = [line.strip() for line in lines]
    lines = [line.split() for line in lines]
    lines = [line for line in lines if len(line) == 12]  
    lines = [[float(i) for i in line] for line in lines]
    data = np.array(lines)
test = data[:, column] / num
mpl.rcParams['font.size'] = 16
plt.figure(figsize=(8, 6))
plt.plot(thermo, 'r-', label='1200K')
plt.plot(test, 'b-',label='300K')
plt.gca().yaxis.set_major_formatter('{:.3f}'.format) 
plt.xlabel('step(100)')  
plt.ylabel('GPa', fontsize=16) 
plt.legend()
plt.tight_layout() 
plt.show()