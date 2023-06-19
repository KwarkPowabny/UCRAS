import numpy as np


func = np.load('Hg_files_S_cut.npy')
R = np.load('Hg_files_R.npy')
derivative = np.zeros((len(func) , len(R)))

print(func[0][0])           
dr=(R[len(R)-1]-R[0])/len(R)
for i in range(len(func)):
 for j in range(len(R)-1):
  df=(func[i][j+1]-func[i][j])/dr
  derivative[i][j] = df

np.save("Hg_files_der_spins_cut", derivative)

