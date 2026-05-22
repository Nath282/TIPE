#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np 
from numpy import pi,cos,sin,sqrt
import matplotlib
matplotlib.use("macosx")
import random as rd
import matplotlib.pyplot as plt

# Paramètres orbitaux 

a = 5.0                    # Demi-grand axe 
e = 0.8                   # Excentricité 
i = np.radians(35)         # Inclinaison compris entre 0 et 180
omega = np.radians(60)     # Argument du périapside compris entre 0 et 360
Omega = np.radians(30)     # Longitude du nœud ascendant compris entre 0 et 360

p = a*(1-e**2)

def SetAxesEqual (ax):
    # Fixe l’échelle des axes 3D pour qu’elle soit égale
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)
    range_max = max(x_range, y_range, z_range) / 2
    ax.set_xlim3d([x_middle - range_max, x_middle + range_max])
    ax.set_ylim3d([y_middle - range_max, y_middle + range_max])
    ax.set_zlim3d([z_middle - range_max, z_middle + range_max])

# Matrice de changement de base : 
    
def R_z (t) : 
    return np.array([[np.cos(t),-np.sin(t),0],
                    [np.sin(t),np.cos(t),0],
                    [0,0,1]])

def R_x (t) : 
    return np.array([[1,0,0],
                    [0,np.cos(t),-np.sin(t)],
                    [0,np.sin(t),np.cos(t)]])

R = R_z(Omega)@R_x(i)@R_z(omega)

def position (theta) : 
    r = p/(1+e*np.cos(theta))
    X = r*np.array([np.cos(theta),np.sin(theta),0])
    return R@X

# Coordonnées dans le referentiel orbital : 
       
theta = np.linspace(0,2*np.pi,500)
r = p/(1+e*np.cos(theta))
x_orb = r * np.cos(theta)
y_orb = r * np.sin(theta)
z_orb = np.zeros_like(x_orb)
xyz_orb = np.array([x_orb,y_orb,z_orb])

# Coordonnées dans le referentiel de reference : 
    
xyz = R @ xyz_orb
x = xyz[0]
y = xyz[1]
z = xyz[2]

# Affichage

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z, label="Ellipse orbitale", color='blue')
ax.scatter([0], [0], [0], color='orange', label="Centre attractif")   

# Axes de référence
ax.quiver(0, 0, 0, 1.5, 0, 0, color='gray', linestyle='dashed', arrow_length_ratio=0.05)
ax.quiver(0, 0, 0, 0, 1.5, 0, color='gray', linestyle='dashed', arrow_length_ratio=0.05)
ax.quiver(0, 0, 0, 0, 0, 1.5, color='gray', linestyle='dashed', arrow_length_ratio=0.05)

ax.text(1.6, 0, 0, 'X', color='gray')
ax.text(0, 1.6, 0, 'Y', color='gray')
ax.text(0, 0, 1.6, 'Z', color='gray')

# Vecteur normal : 
    
u = np.cross(position(0),position(np.pi/2))  
u = u / np.linalg.norm(u)                       # calcul du vecteur normal unitaire
r_min = p/(1+e)                                 # correction de la taille du vecteur
ux,uy,uz = r_min*u[0],r_min*u[1],r_min*u[2] 

ax.quiver(0,0,0,ux,uy,uz,color='red')
    
# Paramètres de la scène

ax.set_title("Ellipse orbitale 3D avec angles orbitaux")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
SetAxesEqual(ax)
ax.set_box_aspect([1,1,1])
#ax.view_init(elev=30, azim=60)
ax.legend()
plt.tight_layout()

plt.show()



















