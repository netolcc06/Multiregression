import numpy as np 
import matplotlib.pyplot as plt
import pptk
import plyfile

def rotX(angle):
    mat = np.array([[1,0,0],[0, np.cos(angle*(np.pi/180)), -np.sin(angle*(np.pi/180))],[0, np.sin(angle*(np.pi/180)), np.cos(angle*(np.pi/180))]])
    return mat

def rotY(angle):
    mat = np.array([[np.cos(angle*(np.pi/180)),0,np.sin(angle*(np.pi/180))],[0, 1, 0],[-np.sin(angle*(np.pi/180)), 0, np.cos(angle*(np.pi/180))]])
    return mat

def rotZ(angle):
    mat = np.array([[np.cos(angle*(np.pi/180)),-np.sin(angle*(np.pi/180)),0],[np.sin(angle*(np.pi/180)), np.cos(angle*(np.pi/180)), 0],[0, 0, 1]])
    return mat

pcd = np.random.rand(1000,3)
armadillo = plyfile.PlyData.read('Truck.ply')['vertex']
#armadillo.shape

xyz = np.c_[armadillo['x'], armadillo['y'], armadillo['z']]
xyz = rotY(180).dot(xyz.T)
xyz = xyz.T
rgb = np.c_[armadillo['red'], armadillo['green'], armadillo['blue']]

v = pptk.viewer(xyz)
v.attributes(rgb / 255.)#, 0.5 * (1 + n))
print(xyz.shape)

print('lets try some point clouds')