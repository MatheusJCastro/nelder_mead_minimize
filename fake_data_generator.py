###################################################
# Fake Cosmological Data Generator                #
# Matheus J. Castro                               #
# Version 1.3                                     #
# Last Modification: 06/11/2021 (month/day/year)  #
###################################################

import numpy as np
from time import gmtime, strftime
import snia

zs = np.loadtxt("SN_2021.cat", skiprows=1).T[0]
errors = np.loadtxt("SN_2021.cat", skiprows=1).T[2]

# Parâmetros dos Dados Falsos
h0 = 70
omega_m = 0.3
omega_ee = 0.7
w = -1
#############################

data = []

for i in zs:
    data.append(snia.lumin_dist_mod_func(h0, i, omega_m, omega_ee, w, show=False, precision=1E-10)[1])

# np.random.seed(int(strftime("%m%d%H%M%S", gmtime())))
# errors = np.abs(np.random.normal(data, np.std(errors), len(errors)))

data_to_save = np.array([zs, data, errors]).T
head = "redshift modulo_de_distancia erro_do_mod_dist"

np.savetxt("fake_data.cat", data_to_save, header=head, fmt="%f")
