import numpy as np
import math as mth
import matplotlib.pyplot as plt


h = 1 
w = 1
m = 1
N = 40
d = 0.01 # x spacing
width = 3*w
x = np.arange(-width, width, d)
length = len(x)
state = np.zeros((N+1, length), dtype=np.complex128)
p = 0

fig, ax = plt.subplots(1,3)
state[0] = np.exp(-m*w/h*x**2)
ax[0].plot(x, state[0])
ax[2].plot(x, state[0])
for n in range(1, N+1):
    norm = (n + 2)**-0.5
    state[n] = state[n-1]
    for i in range(2, length):
        p = (x[i] - x[i-2])*d**-2 # second discreet derivative 
        creationoperator = np.sqrt(m*w/2/h)*x[i] - 1j/np.sqrt(2*m*h*w)*p
        state[n][i] *= norm * np.multiply(state[n][i], creationoperator)
    ax[0].plot(x, state[n].real)
    ax[1].plot(x, state[n].imag)
    ax[2].plot(x, np.absolute(state[0]**2))
norm = (N + 2)**-0.5
state[N] = state[N-1]
for i in range(2, length):
    state[N][i] *= norm*(m*w/2/h)**0.5*x[i]
ax[0].plot(x, state[N].real)
ax[1].plot(x, state[N].imag)
ax[2].plot(x, np.absolute(state[0]**2))
plt.ylabel('Energy')
plt.xlabel('x')
plt.show()

