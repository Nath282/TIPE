#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np 
import random as rd
import matplotlib.pyplot as plt
from time import time 
import math

tol = 1e-8
max_iter = 10

def arrondir_chiffres_significatifs(x, n):
    if x == 0:
        return 0
    else:
        return round(x, -int(math.floor(math.log10(abs(x)))) + (n - 1))


def SolveKepler (M , e) : 
    t0 = time()
    M = M % (2*np.pi)
    #E = InitialisationKepler(M, e)
    if M > np.pi : E = M - e
    else : E = M + e
    n_iter = 0 
    convergence = True
    while True : 
        f = E - e*np.sin(E) - M
        f_prime = 1 - e*np.cos(E)
        delta = f/f_prime 
        E -= delta
        n_iter += 1
        if abs(delta) < tol or n_iter >= max_iter  :
            break
    if n_iter == max_iter :
        convergence = False
    t1 = time()
    return E,n_iter,t1-t0,convergence,prec
    
    
Excentricité = np.linspace(0,0.999,1000)
nbr_res_par_valeur = 10000
N = 0 
N_tot = 0
n_max = 0
t_tot = 0 
Time = np.zeros_like(Excentricité)
Iter = np.zeros_like(Excentricité)
Divergence = np.zeros_like(Excentricité)
prec = []
for e in Excentricité : 
    t_list,n_iter_list = [],[]
    for _ in range(nbr_res_par_valeur) :
        N_tot += 1
        M = rd.uniform(0,2*np.pi)
        _,n_iter,t,convergence,precision = SolveKepler(M,e)
        t_tot += t
        if convergence : 
            t_list.append(t)
            n_iter_list.append(n_iter)
        else : 
            n_max += 1
            prec.append(precision)
            Divergence[N] = Divergence[N] + 1
    Time[N] = np.mean(t_list)
    Iter[N] = np.mean(n_iter_list)
    N += 1

avg_time = t_tot/N_tot
Avg_time = [avg_time]*len(Excentricité)
avg_time = arrondir_chiffres_significatifs(avg_time, 2)
print(tol,max_iter)
print(n_max,avg_time)



# Création de la figure et du premier axe
fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(10, 5))

# Tracer la première courbe
ax1.plot(Excentricité, Time, 'r-', label="temps d'execution")
ax1.plot(Excentricité, Avg_time, 'r:', label="Temps d'execution moyen global = "+str(avg_time)+" s")  # Courbe constante                 
ax1.set_xlabel("Excentricité")
ax1.set_title("Efficacité de KeplerNewtonRaphson ( tol = "+str(tol)+" et max_iter = "+str(max_iter)+")")
ax1.set_ylabel("temps d'execution moyen", color='r')
ax1.legend()
ax1.tick_params(axis='y', labelcolor='r')

# Création du second axe partageant le même axe x
ax2 = ax1.twinx()
ax2.plot(Excentricité, Iter, 'b-', label="Nombre d'iterations")
ax2.set_ylabel("nombre d'iterations moyen par valeur de e ", color='b')
ax2.tick_params(axis='y', labelcolor='b')

# --- Graphique de droite : une seule courbe ---
ax3.plot(Excentricité, Divergence, 'g', label='Nombre de divergence , total = ' + str(n_max) + ' sur ' + str(N_tot))
ax3.set_title("Convergence de KeplerNewtonRaphson")
ax3.set_xlabel('Excentricité')
ax3.set_ylabel('Nombre de divergence (sur '+str(nbr_res_par_valeur)+' essais par valeurs)')
ax3.legend()

# Améliorer l'espacement
plt.tight_layout()
plt.show()





    
    
    
    
    
    
    
    
    
    
    
    
    
    
    