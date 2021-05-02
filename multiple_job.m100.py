# Simple Python script for submitting many jobs in parallel running the Monte Carlo code at different temperatures on different nodes

import os
import string
import numpy as np

t=(1.00e-07,1.00e-05,1.00e-03)
npoint = len(t)

delta00 = {1.00e-07: 6e-04, 5.00e-07: 6e-03, 1.00e-06: 9e-03, 2.5e-06: 0.02, 5e-06: 0.03, 1.00e-05: 0.04, 
2.5e-05: 0.1, 5e-05: 0.2, 1.00e-04: 0.3, 5e-04: 0.4, 1.00e-03: 0.5, 5e-03: 0.7}
delta20 = {1.00e-07: 3.6e-06, 5.00e-07: 9.5e-05, 1.00e-06: 6e-05, 2.5e-06: 9e-05, 5e-06: 2e-04, 1.00e-05: 5e-04,
2.5e-05: 9e-04, 5e-05: 2e-03, 1.00e-04: 5e-03, 5e-04: 0.02, 1.00e-03: 0.04, 5e-03: 0.06}
delta50 = {1.00e-07: 1.00e-06, 5.00e-07: 6e-06, 1.00e-06: 1.00e-06, 2.5e-06: 3e-05, 5e-06: 6e-05, 1.00e-05: 1.5e-04,
2.5e-05: 4.00e-04, 5e-05: 6e-04, 1.00e-04: 1.5e-03, 5e-04: 6e-03, 1.00e-03: 0.013, 5e-03: 0.018}
deltaR = {1.00e-07: 0.1, 5.00e-07:0.4, 1.00e-06:0.45, 2.5e-06: 0.75, 5e-06: 0.95, 1.00e-05: 1.2, 
2.5e-05: 1.5, 5e-05: 1.57, 1.00e-04: 1.57, 5e-04: 1.57, 1.00e-03: 1.57, 5e-03: 1.57}
deltaB = {1.00e-07: 5.5e-07, 5.00e-07: 7e-06, 1.00e-06: 7.5e-06, 2.5e-06: 2e-05,5e-06: 3.5e-05, 1.00e-05: 7e-05, 2.5e-05: 2e-04,
5e-05: 3.5e-04, 1.00e-04: 7e-04, 5e-04: 3e-03, 1.00e-03: 6e-03, 5e-03: 8e-03}
deltaCM = {1.00e-07: 3e-04,5.00e-07: 4e-04, 1.00e-06: 8e-04, 2.5e-06: 1.5e-03, 5e-06: 1.8e-03, 1.00e-05: 3e-03, 2.5e-05: 3.5e-03, 
5e-05: 6e-03, 1.00e-04: 9e-03,5e-04: 1.8e-02, 1.00e-03: 2.4e-02, 5e-03: 3e-02}

for i in range(npoint):
	t_name = round(t[i], -int(np.log10(t[i]))+3)
	path = 'T_{:.2e}'.format(t_name)
	if not os.path.exists(path):
		os.mkdir(path)
	os.system('cp original_code/* '+ path)

	s = open(path + '/input.dat').read()
	s = s.replace('$TEMP', format(t[i])).replace('$DELTA00', format(delta00[t[i]])).replace('$DELTA20', format(delta20[t[i]]))..replace('$DELTA50', format(delta50[t[i]])).replace('$DELTAR', format(deltaR[t[i]])).replace('$DELTAB', format(deltaB[t[i]])).replace('$DELTACM', format(deltaCM[t[i]]))
	f = open(path + '/input.dat', 'w')
	f.write(s)
	f.close()
	
	s =open(path+ '/mpijob.m100').read()
	s = s.replace('$TEMP', format(t_name))
	f = open(path + '/mpijob.m100', 'w')
	f.write(s)
	f.close()

	os.chdir(path)
	os.system('make clean')
	os.system('make clear')
	os.system('make')
	os.system('sbatch mpijob.m100')
	os.chdir('..')
