# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 12:17:49 2023

@author: Pawel


using altered Hg_energy_func
"""
import numpy as np
import matplotlib.pyplot as plt
import Hg_energy_func as func

crossings = np.load("crossings")
ok_values = np.load("another file")
spins_derivative = np.load("f")
state_derivative = np.load("f")


for r in range(first_crossing - 3, last_crossing + 3):
    for i in range(num_states):
        if (... or ...)