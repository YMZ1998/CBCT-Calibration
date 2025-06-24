import importlib
import os

import matplotlib.pyplot as plt
import numpy as np

import amglib.readers as io
import cbct.CBCT_Calibration as cbct

datapath = r'./data/P20240130_1/cbct_cal/'
# print(os.listdir(datapath))
# dc = io.read_images(datapath + 'dc_{:05}.fits', first=1, last=5, averageStack=True, average='mean')
# ob = io.read_images(datapath + 'ob_{:05}.fits', first=1, last=5, averageStack=True, average='mean')
# proj = io.read_images(datapath + 'ct_{:05}.fits', first=1, last=360, stride=10)
# np.save("dc.npy", dc)
# np.save("ob.npy", ob)
# np.save("proj.npy", proj)

dc = np.load("dc.npy")
ob = np.load("ob.npy")
proj = np.load("proj.npy")

importlib.reload(cbct)
cal = cbct.CBCTCalibration()
cal.set_projections(proj=proj, ob=ob, dc=dc, verticalflip=True, show=True)

cal.remove_projection_baseline(show=True)

cal.flatten_projections(amplification=10, stack_window=5, show=True)

cal.threshold_projections(threshold=0.1, show=True, clearborder=True)

cal.find_beads(breakup=False, show=True)
cal.find_trajectories(show=True)
cal.fit_ellipses(show=True, prune=True)
e = cal.ellipses
print(e)

# importlib.reload(cbct)
#
# cal = cbct.CBCTCalibration()
# cal.ellipses = e
cal.compute_calibration(diameter=20, avgtype='mean', remove_outliers=True, show=True)

print(cal.calibration)
plt.show()
