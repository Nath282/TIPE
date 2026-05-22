#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Iterator
import numpy as np 
from numpy import pi,cos,sin,sqrt,array
from numpy.linalg import norm
import matplotlib
matplotlib.use("macosx")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D as ax
import random as rd
from itertools import combinations # pour pouvoir generer les couples non ordonnés d'un set 

import logging
logging.basicConfig(
    level=logging.INFO,  # Niveau de base (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(levelname)s - %(message)s',  # Format du message
    filename = 'animation2.log',  # Fichier de sortie (ou None pour console)
    filemode='w'  # 'w' pour écraser à chaque exécution, 'a' pour ajouter
)

"""
# PARAMETRES DU SYSTEME #
"""

G = 1        # Constante de gravitation
M = 1        # Masse de l'astre (masse unitaire)
R = 0.355    # Rayon de l'astre 
mu = G * M

n = 20      # Nombre de particules
m = 2*n      # Nombre de particule maximal du système 
rho = 7.12   # masse volumique de chaque particule du système à l'échelle 

R1 = 1       # Periastre minimal de la distribution (longueur unitaire defini comme égale à la limite de Roche)
R2 = 2     # Apoastre maximal de la distribution

radius = 0.023*R

beta = 1.5   # paramètre d'elasticité des chocs (compris dans [1,2])

global_t = 0                    # instant initial
pas_discretisation = 100  # pas de discrétisation du temps
delta_t = 2 * pi * sqrt( R1**3 / mu ) / pas_discretisation 
T = 40                    # Periode de cycle long
NbrCycleLong = 3         # Nombre maximal de cycle long effectué par le système 

CollisionCounter = 0 
MaxCollision = 1000

tol_NR = 1e-8   # precision demandé pour résoudre l'equation de Kepler 
max_iter = 10   # nombre d'iterations maximal pour Newton-Raphson

nbr_points = 10   # nombre de points de l'echantillonage lors du filtrage spatial
tol_filtrage = R  # tolerance du filtrage spatial 


tol_Taylor = 1e-8  # precision demandé sur l'instant de la collision


"""
# Resolution numerique de l'equation de Kepler
"""

def KeplerNewtonRaphson (M,e) :
    if M > pi : E = M - e  # Initialisation de Newton-Raphson
    else : E = M + e
    n_iter,convergence = 0,True 
    while True : 
        f = E - e*sin(E) - M 
        f_prime = 1 - e*cos(E)
        delta = f/f_prime
        E -= delta
        n_iter += 1
        if abs(delta) < tol_NR :
            break
        if n_iter == max_iter - 1 :
            convergence = False
            break
    return E,convergence

def KeplerDichotomie (M,e) :
    E_min,E_max = 0,2*pi
    max_iter = 2 + int(np.log2(pi/tol_NR)) # nombre d'itérations pour obtenir une precision suffisante -> 30 pour 1e-8
    f = lambda E : E - e*sin(E) - M
    for _ in range(max_iter) :
        E_middle = (E_max+E_min)/2
        if f(E_min)*f(E_middle) < 0 :
            E_max = E_middle
        else : 
            E_min = E_middle
    return (E_max+E_min)/2

def SolveKepler (M,e) : 
    M = np.mod(M,2*pi)
    E,convergence = KeplerNewtonRaphson(M, e)
    if convergence : 
        return E 
    else : 
        return KeplerDichotomie(M, e)
    

"""
# Fonctions auxiliaires de selection aleatoire des paramètres orbitaux 
"""

def RandomConicParameters () : 
    # Selection aléatoire des paramètres de l'ellipse
    # selection tel que la trajectoire soit comprise entre les sphères de rayon R1 et R2
    a = rd.uniform(R1,R2)                # demi-grand axe a
    e = rd.uniform(0,0.999)              # excentricité e
    while a*(1-e) < R1 or a*(1+e) > R2 : 
        a = rd.uniform(R1,R2)            
        e = rd.uniform(0,0.999)
    return a,e

def RandomEulerAngles () : 
    # Selection aléatoire des angles d'Euler paramètrant la trajectoire en 3D
    # l'inclinaison est choisie tel que cos(i) soit uniforme dans [-1,1]
    i = np.arccos(rd.uniform(-1,1))   # inclinaison i
    O = rd.uniform(0,2*pi)            # longitude du noeud ascendant O
    w = rd.uniform(0,2*pi)            # argument du periastre w
    return i,O,w

"""
# Definition de la classe OrbitalElements
"""

class OrbitalElements :
    
    def __init__ (self,a=None,e=None,i=None,O=None,w=None,E0=None,r=None) :
        if not a or not e or not i or not O or not w or not E0 : 
            a,e = RandomConicParameters()
            i,O,w = RandomEulerAngles()
            E0 = rd.uniform(0,2*pi)  # Anomalie excentrique initiale  
            r = radius
        self.a,self.e,self.i,self.O,self.w,self.E0,self.radius = a,e,i,O,w,E0,r
        self._ComputeOrbitParameters()
        self._ComputeOrbitRotationalMatrix()
        
    def _ComputeOrbitParameters (self) :
        self.M0 = self.E0 - self.e * sin( self.E0 ) # Anomalie moyenne initiale M0 
        self.mass = 4/3 * pi * rho * self.radius**3 # masse 
        self.T = 2 * pi * sqrt( self.a**3 / mu )    # Période
        self.p = self.a * ( 1 - self.e**2 )         # Semiparamètre
        self.b = self.a * sqrt( 1 - self.e**2 )     # Demi-petit axe
        h = (self.a-self.b)**2 / (self.a+self.b)**2 
        self.P = pi * (self.a+self.b) * (3-sqrt(4-h)) # Perimetre (1ere formule de Ramanujan)
        
    def _ComputeOrbitRotationalMatrix (self) :
        cos_i,sin_i = cos(self.i),sin(self.i)   
        cos_O,sin_O = cos(self.O),sin(self.O)
        cos_w,sin_w = cos(self.w),sin(self.w)
        self.R = array([[ cos_O*cos_w - sin_O*sin_w*cos_i , - cos_O*sin_w - sin_O*cos_w*cos_i , sin_O*sin_i ],
                          [ sin_O*cos_w + cos_O*sin_w*cos_i , - sin_O*sin_w + cos_O*cos_w*cos_i , - cos_O*sin_i ],
                          [ sin_w*sin_i , cos_w*sin_i , cos_i ]])

    @property        
    def Periastre (self) : return self.a * ( 1 - self.e ) 
    @property
    def Apoastre (self) : return self.a * ( 1 + self.e )
    
    def CheckOrbit (self) : 
        if self.e >= 1 or self.Periastre + self.radius < R : return False 
        else : return True

    def _UpdateOrbitalElements (self,a,e,i,O,w,M0) : 
        self.a,self.e,self.i,self.O,self.w,self.M0 = a,e,i,O,w,M0
        self._ComputeOrbitParameters
        self._ComputeOrbitRotationalMatrix()
            

"""
# Definition de la classe Particle héritant de la classe OrbitalElements
"""  
    
class Particle (OrbitalElements) :
    
    def __init__ (self,id,a=None,e=None,i=None,O=None,w=None,E0=None,r=None) :
        self.id = id  # Warning : paramètre immuable qui ne doit pas être modifié 
        self.version = 0 
        self.t_collision = global_t
        super().__init__(a,e,i,O,w,E0,r)  # initialise les éléments orbitaux de la particule
        
    def __eq__ (self,other) : 
        # Warning : deux particules peuvent être les mêmes mais de version differentes
        return isinstance(other, Particle) and self.id == other.id

    def __hash__ (self) : 
        # assure l'utilisation dans un set
        return hash(self.id)
    
    def UpdateOrbit (self,a,e,i,O,w,M0,t) : 
        self._UpdateOrbitalElements(a,e,i,O,w,M0)
        self.version += 1
        self.t_collision = t
    
    def AnomalieExcentrique (self,t) :
        n = 2 * pi / self.T
        M = np.mod(self.M0 + n * (t-self.t_collision) , 2 * pi)
        E = SolveKepler(M, self.e) # calcul de l'anomalie excentrique en resolvant l'equation de Kepler
        return np.mod(E, 2 * pi)
    
    def OrbitalRadius (self,E) : return self.a * ( 1 - self.e*cos(E))  # cf p75
        
    def Position (self,E) : # Calcul du vecteur position 
        X_inertiel = self.a * array([cos(E)-self.e, sqrt(1-self.e**2)*sin(E), 0 ]) # cf Trulsen I
        return self.R @ X_inertiel
    
    def Speed (self,E) : # Calcul du vecteur vitesse
        V_inertiel = array([-sin(E), sqrt(1-self.e**2)*cos(E), 0 ]) / (sqrt(self.a)*(1-self.e*cos(E)))             
        return self.R @ V_inertiel
    
    def Acceleration (self,E) : # Calcul du vecteur acceleration
        A_inertiel = - mu / self.OrbitalRadius(E)**3 * self.a * array([cos(E)-self.e, sqrt(1-self.e**2)*sin(E), 0 ])
        return self.R @ A_inertiel
    
    
"""
# Procédure de collision
"""

def BetaModel (p1 : Particle,p2 : Particle, E1, E2) :
    r1, r2 = p1.Position(E1), p2.Position(E2)
    v1, v2 = p1.Speed(E1), p2.Speed(E2)
    m1, m2 = p1.mass, p2.mass
    k = (r1-r2)/np.linalg.norm(r1-r2)  # Vecteur de collision
    alpha = np.dot(v1-v2, k)
    new_v1 = v1 - m2 * beta * alpha / (m1 + m2) * k
    new_v2 = v2 + m1 * beta * alpha / (m1 + m2) * k
    return new_v1,new_v2

def ComputeOrbitalElements (p,R,V) : 
    # Calcul des parametres orbitaux à partir des conditions initiales (cf p 114)
    # Les vecteurs sont representés en majuscules et les scalaires en minuscule
    r, v = np.linalg.norm(R), np.linalg.norm(V)
    K = array([0,0,1])                                 # vecteur e_z
    H = np.cross(R, V)                                    # Moment angulaire spécifique   
    h = np.linalg.norm(H)                                 # constante des aires
    N = np.cross(K, H)
    n = np.linalg.norm(N)
    E = ( v**2 - mu / r ) / mu * R - np.dot(R, V)/mu * V  # vecteur excentricité
    e = np.linalg.norm(E)                                 # excentricité 
    epsilon = v**2 / 2 - mu / r
    a = - mu/( 2 * epsilon )                              # demi-grand axe 
    i = np.arccos(H[2] / h)                               # inclinaison
    Omega = np.arccos(N[0] / n)                           # longitude du noeud ascendant 
    if N[1] < 0 :
        Omega = 2 * pi - Omega
    omega = np.arccos( np.dot(N, E) / ( n * e ) )         # argument du périastre 
    if E[2] < 0 : 
        omega = 2 * pi - omega
    nu = np.arccos( np.dot(E, R) / ( e * r ) )            # anomalie vraie 
    if np.dot(R, V) < 0 :
        nu = 2 * pi - nu
    sin_E = sin(nu) * sqrt( abs(1 - e**2) ) / ( 1 + e * cos(nu) ) # cf p77 (valeur absolue au cas où e > 1 qui sera invalidé plus tard)
    cos_E = ( e + cos(nu) ) / ( 1 + e * cos(nu) )
    E0 = np.arctan2(sin_E, cos_E)                          # anomalie excentrique 
    return (a,e,i,Omega,omega,E0)

def ProcessCollision (p1,p2,E1,E2) :
    r1, r2 = p1.Position(E1), p2.Position(E2)
    v1, v2 = BetaModel(p1, p2, E1, E2)
    a1,e1,i1,O1,w1,M01 = ComputeOrbitalElements(p1, r1, v1)
    p1.UpdateOrbitalElements(a1,e1,i1,O1,w1,M01)
    a2,e2,i2,O2,w2,M02 = ComputeOrbitalElements(p2, r2, v2)
    p2.UpdateOrbitalElements(a2,e2,i2,O2,w2,M02)
    

"""
# Detection de collision
"""

def DichoTaylor (p1 : Particle,p2 : Particle, E1, E2, t, delta_t) : 

    RadiusSum = p1.radius + p2.radius
    r1, r2 = p1.Position(E1), p2.Position(E2)
    v1, v2 = p1.Speed(E1), p2.Speed(E2)
    a1, a2 = p1.Acceleration(E1), p2.Acceleration(E2)
    d = lambda h : np.linalg.norm( r1-r2 + h*(v1-v2) + 0.5*h**2*(a1-a2) ) - RadiusSum # developpement limité autour de t_min     
    
    t_inf = -delta_t   # bornes pour la dichotomie
    if d(t_inf) < 0 : t_inf = -2*delta_t # cas ou on a une difference entre la distance et son developpement limité 
    t_sup = 0  

    nbr_iter = 1 + int(np.log2(t_sup-t_inf/tol_Taylor)) # nombre d'iterations pour obtenir la precision demandé
    
    for _ in range(nbr_iter) :
        t_middle = (t_sup+t_inf)/2
        if d(t_inf)*d(t_middle) < 0 :
            t_sup = t_middle
        else : 
            t_inf = t_middle
            
    return t+(t_sup+t_inf)/2
 
def Distance (p1 : Particle, p2 : Particle, E1, E2) :
    SumRadius = p1.radius + p2.radius
    return np.linalg.norm( p1.Position(E1) - p2.Position(E2) ) - SumRadius

def Perimetre (p : Particle) :
    a, b = p.a, p.b 
    h = (a-b)**2 / (a+b)**2 
    return pi * (a+b) * (3-sqrt(4-h)) 

def ComputeStep (p1 : Particle, p2 : Particle, E1, E2, local_tol=0.5, orbital_tol=0.01) : 
    R1, R2 = p1.radius, p2.radius
    v1, v2 = norm(p1.Speed(E1)), norm(p2.Speed(E2))
    d = Distance(p1, p2, E1, E2)
    R_sum = R1 + R2
    AvgDist = abs(p1.a -p2.a) + abs(p1.e - p2.e) * (p1.a + p2.a)/2

    local_dt = local_tol * min(R1 / v1, R2 / v2) * (d / R_sum )
    orbital_dt = orbital_tol * min(p1.P / v1, p2.P / v2) * (d / AvgDist)

    dt_min = 0.2 * min(R1 / v1, R2 / v2)
    dt_max = 0.1 * min(p1.P / v1, p2.P / v2)

    if d < 100 * R_sum : # echelle locale 
        logging.info("delta_t local = %s", delta_t)
        return max(local_dt, dt_min)
    
    else : # echelle orbitale
        logging.info("delta_t orbital = %s", delta_t)
        return min(orbital_dt, dt_max)
    



"""
# Affichage
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

def AffichageEllipse (ax,p,couleur) :
    EList = np.linspace(0,2*np.pi,50)
    x,y,z = np.array([ p.Position(E) for E in EList ]).T
    line, = ax.plot(x, y, z, color=couleur) # permet de retourner la courbe et pas l'ensemble des points
    return line

def AffichageSphere (ax,centre,rayon,couleur,transparence) : 
    theta = np.linspace(0, 2 * np.pi, 20)
    phi = np.linspace(0, np.pi, 10)
    theta, phi = np.meshgrid(theta, phi)
    x = centre[0] + rayon * np.sin(phi) * np.cos(theta)
    y = centre[1] + rayon * np.sin(phi) * np.sin(theta)
    z = centre[2] + rayon * np.cos(phi)
    return ax.plot_surface(x, y, z, color=couleur, alpha=transparence)

def AffichageVecteur (ax,point,vecteur,couleur) :
    x,y,z = point
    v_x,v_y,v_z = vecteur
    return ax.quiver(x,y,z,v_x,v_y,v_z,color=couleur)

def InitAffichage (ax,p1,p2,transparence) :
    # Affichage des deux orbites
    ellipse1 = AffichageEllipse(ax,p1,'red')
    ellipse2 = AffichageEllipse(ax,p2,'red')

    # Affichage de Saturne 
    Saturne = AffichageSphere(ax, [0,0,0], R, 'orange',0.9)

    # Affichage initiale des sphères 
    E1, E2 = p1.AnomalieExcentrique(global_t), p2.AnomalieExcentrique(global_t)
    sph1 = AffichageSphere(ax, p1.Position(E1), p1.radius, 'grey', transparence)
    sph2 = AffichageSphere(ax, p2.Position(E2), p2.radius, 'grey', transparence)

    # Affichage initiale des vecteurs vitesse 
    quiv1 = AffichageVecteur(ax, p1.Position(E1), p1.Speed(E1), 'red')
    quiv2 = AffichageVecteur(ax, p2.Position(E2), p2.Speed(E2), 'red')
    return ellipse1, ellipse2, sph1, sph2, quiv1, quiv2

"""
# Animation
"""

p1 = Particle(0)
p2 = Particle(1)

logging.info("Particule 1 de periode T = %s",p1.T)
logging.info("Particule 2 de periode T = %s",p2.T)

# en cas de collision initial regénère des particules
while Distance(p1, p2, p1.E0, p2.E0) <= 0 :
    p1 = Particle(0)
    p2 = Particle(1)

# Initialisation de l'animation    
plt.ion()
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
transparence = 0.3

ellipse1, ellipse2, sph1, sph2, quiv1, quiv2 = InitAffichage (ax,p1,p2,transparence)

# Initialisation de la detection
collision = False 
t = 0
    
while t < T and not collision : 

    E1,E2 = p1.AnomalieExcentrique(t),p2.AnomalieExcentrique(t)

    # Retrait des sphères précédentes et ajout des nouvelles
    sph1.remove()
    sph2.remove()
    sph1 = AffichageSphere(ax, p1.Position(E1) , p1.radius, 'grey', transparence) 
    sph2 = AffichageSphere(ax, p2.Position(E2) , p2.radius, 'grey', transparence)
    
    # Retrait des anciens vecteurs vitesses et ajout des nouveaux 
    quiv1.remove()
    quiv2.remove()
    quiv1 = AffichageVecteur(ax, p1.Position(E1), p1.Speed(E1), 'red')
    quiv2 = AffichageVecteur(ax, p2.Position(E2), p2.Speed(E2), 'red')

    # Boucle de detection
    d = Distance(p1, p2, E1, E2)
    delta_t = ComputeStep(p1, p2, E1, E2)

    if d <= 0 :
        collision = True
        t1 = t
        t = DichoTaylor(p1, p2, E1, E2, t, delta_t) 
        E1,E2 = p1.AnomalieExcentrique(t),p2.AnomalieExcentrique(t) 
        d2 = Distance(p1, p2, E1, E2)
        logging.info("distance après Taylor %s au temps %s",d2,t)
        logging.info("distance detection collision %s au temps %s",d,t1)
        r1, r2 = p1.Position(E1), p2.Position(E2)
        v1,v2 = BetaModel(p1, p2, E1, E2)

        a1,e1,i1,O1,w1,M01 = ComputeOrbitalElements(p1, r1, v1)
        p1.UpdateOrbit(a1,e1,i1,O1,w1,M01,t)
        a2,e2,i2,O2,w2,M02 = ComputeOrbitalElements(p2, r2, v2)
        p2.UpdateOrbit(a2,e2,i2,O2,w2,M02,t)

        # Ajout des nouvelles trajectoires
        AffichageEllipse(ax,p1,'blue')
        AffichageEllipse(ax,p2,'blue')

        # Ajout des sphères aux nouvelles conditions initiales
        E1, E2 = SolveKepler(p1.M0, p1.e), SolveKepler(p2.M0, p2.e)
        r1, r2 = p1.Position(E1), p2.Position(E2)
        AffichageSphere(ax, r1 , p1.radius, 'grey', transparence) 
        AffichageSphere(ax, r2 , p2.radius, 'grey', transparence)
        AffichageVecteur(ax, r1, v1, 'blue')
        AffichageVecteur(ax, r2, v2, 'blue')

        # Affichage 
        SetAxesEqual(ax)
        ax.set_title("Animation 2 particules")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.tight_layout()
        plt.pause(0.05)

    else : 
        t += delta_t 

    """
    if t >= t0 + T and not collision : 
        ax.cla()
        t = t0
        p1 = Particle(0)
        p2 = Particle(1)
        ellipse1, ellipse2, sph1, sph2, quiv1, quiv2 = InitAffichage (ax,p1,p2,transparence)
    """

    # Affichage 
    SetAxesEqual(ax)
    ax.set_title("Animation 2 particules")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    #ax.view_init(elev=30, azim=60)
    plt.tight_layout()
    plt.pause(0.05)

plt.ioff()
plt.show()





























