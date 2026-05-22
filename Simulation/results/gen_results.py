#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib
matplotlib.use("macosx")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D as ax
from numpy import pi, sin, cos, sqrt, array
import numpy as np

df = pd.read_csv("TIPE/Simulation/data/donnees_sim.csv")

# Trouver la première valeur de t
premier_t = df["t"].min()
# Extraire les lignes correspondant à ce t
premier_etat = df[df["t"] == premier_t]

# Trouver la dernière valeur de t
dernier_t = df["t"].max()
# Extraire les lignes correspondant à ce t
dernier_etat = df[df["t"] == dernier_t]


df_stats = df.groupby("t")[["a", "e", "i", "O", "w", "m"]].agg(["mean", "std", "sum"])
# Aplatir l'index de colonnes
df_stats.columns = ['_'.join(col) for col in df_stats.columns]
df_stats = df_stats.reset_index()


"""
# Fonctions auxiliaires
"""

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

def Affichage (ax,a,e,i,O,w) : 
    R = array([[ cos(O) * cos(w) - sin(O)*sin(w)*cos(i) , - cos(O)*sin(w) - sin(O)*cos(w)*cos(i) , sin(O)*sin(i) ],
                          [ sin(O)*cos(w) + cos(O)*sin(w)*cos(i) , - sin(O)*sin(w) + cos(O)*cos(w)*cos(i) , - cos(O)*sin(i) ],
                          [ sin(w)*sin(i) , cos(w)*sin(i) , cos(i) ]])
    r = lambda E : R @ (a * array([cos(E)-e, sqrt(1-e**2)*sin(E), 0]))
    EList = np.linspace(0,2*np.pi,40)
    x,y,z = np.array([ r(E) for E in EList ]).T
    ax.plot(x, y, z, color='blue', linewidth = 0.25) 

def AffichageSphere (ax,centre,rayon,couleur,nbr_points) : 
    theta = np.linspace(0, 2 * np.pi, 2*nbr_points)
    phi = np.linspace(0, np.pi, nbr_points)
    theta, phi = np.meshgrid(theta, phi)
    x = centre[0] + rayon * np.sin(phi) * np.cos(theta)
    y = centre[1] + rayon * np.sin(phi) * np.sin(theta)
    z = centre[2] + rayon * np.cos(phi)
    return ax.plot_surface(x, y, z, color=couleur, alpha=0.8)


"""
# Tracé des courbes
"""


plt.figure(figsize=(12, 8))


# Demi-grand axe :
plt.subplot(2, 3, 1)
plt.plot(df_stats["t"], df_stats["a_mean"], label="a moyen")
plt.fill_between(df_stats["t"],
                 df_stats["a_mean"] - df_stats["a_std"],
                 df_stats["a_mean"] + df_stats["a_std"],
                 color="blue", alpha=0.3, label="±écart_type")

plt.ylabel("Demi-grand axe (a)")
plt.title("Évolution de a avec incertitude")
plt.legend()
plt.grid(True)


# Excentricité : 
plt.subplot(2, 3, 2)
plt.plot(df_stats["t"], df_stats["e_mean"], label="e moyen")
plt.fill_between(df_stats["t"],
                 df_stats["e_mean"] - df_stats["e_std"],
                 df_stats["e_mean"] + df_stats["e_std"],
                 color="blue", alpha=0.3, label="±écart_type")

plt.ylabel("Excentricité (e)")
plt.title("Évolution de e avec incertitude")
plt.legend()
plt.grid(True)


# Inclinaison : 
plt.subplot(2, 3, 3)
plt.plot(df_stats["t"], df_stats["i_mean"], label="i moyen")
plt.fill_between(df_stats["t"],
                 df_stats["i_mean"] - df_stats["i_std"],
                 df_stats["i_mean"] + df_stats["i_std"],
                 color="blue", alpha=0.3, label="±écart_type")

plt.ylabel("Inclinaison (e)")
plt.title("Évolution de i avec incertitude")
plt.legend()
plt.grid(True)


# Longitude du noeud ascendant : 
plt.subplot(2, 3, 4)
plt.plot(df_stats["t"], df_stats["O_mean"], label="O moyen")
plt.fill_between(df_stats["t"],
                 df_stats["O_mean"] - df_stats["O_std"],
                 df_stats["O_mean"] + df_stats["O_std"],
                 color="blue", alpha=0.3, label="±écart_type")

plt.xlabel("Temps")
plt.ylabel("Longitude du noeud ascendant (O)")
plt.title("Évolution de O avec incertitude")
plt.legend()
plt.grid(True)


# Argument du periastre : 
plt.subplot(2, 3, 5)
plt.plot(df_stats["t"], df_stats["w_mean"], label="w moyen")
plt.fill_between(df_stats["t"],
                 df_stats["w_mean"] - df_stats["w_std"],
                 df_stats["w_mean"] + df_stats["w_std"],
                 color="blue", alpha=0.3, label="±écart_type")

plt.xlabel("Temps")
plt.ylabel("Argument du periastre")
plt.title("Évolution de w avec incertitude")
plt.legend()
plt.grid(True)


# Masse : 
plt.subplot(2, 3, 6)

df_filtered = df_stats[df_stats["t"] > 0]

plt.plot(df_filtered["t"], df_filtered["m_sum"], label="m totale", color='red')
plt.xlabel("Temps")
plt.title("Évolution de m_tot avec incertitude")
plt.legend()
plt.grid(True)


plt.savefig("Resultats.pdf")


"""
# Affichage du système à l'état initial
"""

# Colonnes à extraire
colonnes_orbitales = ["a", "e", "i", "O", "w"]
# Conversion en tableau numpy (shape: [nb_particules, 5])
T = premier_etat[colonnes_orbitales].to_numpy()


fig = plt.figure()
ax = fig.add_subplot(111, projection="3d") 

R = 0.355
AffichageSphere(ax, [0,0,0], R, 'orange',100)
for orbital_elements in T : 
    [a,e,i,O,w] = orbital_elements
    Affichage(ax,a,e,i,O,w)

ax.set_title("Système initial")
SetAxesEqual(ax)

plt.savefig("Système initial.pdf")


"""
# Affichage du système à l'etat final 
"""

# Colonnes à extraire
colonnes_orbitales = ["a", "e", "i", "O", "w"]
# Conversion en tableau numpy (shape: [nb_particules, 5])
T = dernier_etat[colonnes_orbitales].to_numpy()


fig = plt.figure()
ax = fig.add_subplot(111, projection="3d") 

R = 0.355
AffichageSphere(ax, [0,0,0], R, 'orange',100)
for orbital_elements in T : 
    [a,e,i,O,w] = orbital_elements
    Affichage(ax,a,e,i,O,w)

ax.set_title("Système final")
SetAxesEqual(ax)

plt.savefig("Système final.pdf")












