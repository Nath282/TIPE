#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np 
from numpy import pi,cos,sin,sqrt,array
import matplotlib
matplotlib.use("macosx")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D as ax
import random as rd
from itertools import combinations # pour pouvoir generer les couples non ordonnés d'un set 

"""
# PARAMETRES DU SYSTEME #
"""

G = 1        # Constante de gravitation
M = 1        # Masse de l'astre (masse unitaire)
R = 0.355    # Rayon de l'astre 
mu = G * M

n = 10      # Nombre de particules
m = 2*n      # Nombre de particule maximal du système 
rho = 7.12   # masse volumique de chaque particule du système à l'échelle 

R1 = 1       # Periastre minimal de la distribution (longueur unitaire defini comme égale à la limite de Roche)
R2 = 1.5       # Apoastre maximal de la distribution

radius = R/sqrt(2)

beta = 1.5   # paramètre d'elasticité des chocs (compris dans [1,2])

t0 = 0                    # instant initial
pas_discretisation = 100  # pas de discrétisation du temps
delta_t = 2 * pi * sqrt( R1**3 / mu ) / pas_discretisation 
T = 50                    # Periode de cycle long
NbrCycleLong = 5         # Nombre maximal de cycle long effectué par le système 

CollisionCounter = 0 
MaxCollision = 1000

tol_NR = 1e-8   # precision demandé pour résoudre l'equation de Kepler 
max_iter = 10   # nombre d'iterations maximal pour Newton-Raphson

nbr_points = 10   # nombre de points de l'echantillonage lors du filtrage spatial
tol_filtrage = R  # tolerance du filtrage spatial 


tol_Taylor = 1e-8  # precision demandé sur l'instant de la collision

CollisionQueue = []


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
    
    def __init__ (self,a=None,e=None,i=None,O=None,w=None,M0=None,r=None) :
        if not a or not e or not i or not O or not w or not M0 : 
            a,e = RandomConicParameters()
            i,O,w = RandomEulerAngles()
            M0 = rd.uniform(0,2*pi)    # Anomalie moyenne initiale M0 
            r = radius
        self.a,self.e,self.i,self.O,self.w,self.M0,self.radius = a,e,i,O,w,M0,r
        self._ComputeOrbitParameters()
        self._ComputeOrbitRotationalMatrix()
        
    def _ComputeOrbitParameters (self) :
        self.mass = 4/3 * pi * rho * self.radius**3 # masse 
        self.T = 2 * pi * sqrt( self.a**3 / mu )    # Période
        self.p = self.a * ( 1 - self.e**2 )         # Semiparamètre
        self.C = sqrt( self.p * mu )                # Constante des aires (ou moment angulaire spécifique)
        
    def _ComputeOrbitRotationalMatrix (self) :
        cos_i,sin_i = cos(self.i),sin(self.i)   
        cos_O,sin_O = cos(self.O),sin(self.O)
        cos_w,sin_w = cos(self.w),sin(self.w)
        self.R = array([[ cos_O*cos_w - sin_O*sin_w*cos_i , - cos_O*sin_w - sin_O*cos_w*cos_i , sin_O*sin_i ],
                          [ sin_O*cos_w + cos_O*sin_w*cos_i , - sin_O*sin_w + cos_O*cos_w*cos_i , - cos_O*sin_i ],
                          [ sin_w*sin_i , cos_w*sin_i , cos_i ]])
            
    def Periastre (self) : return self.a * ( 1 - self.e ) 
    def Apoastre (self) : return self.a * ( 1 + self.e )
    
    def CheckOrbit (self) : 
        if self.e > 1 or self.e < 0 or self.Periastre() + self.radius < R : return False 
        else : return True
        
    def _UpdateOrbitalElements (self,a,e,i,O,w,M0) : 
        self.a,self.e,self.i,self.O,self.w,self.M0 = a,e,i,O,w,M0
        self._ComputeOrbitParameters
        self._ComputeOrbitRotationalMatrix()
            

"""
# Definition de la classe Particle héritant de la classe OrbitalElements
"""  
    
class Particle (OrbitalElements) :
    
    def __init__ (self,id,a=None,e=None,i=None,O=None,w=None,M0=None,r=None) :
        self.id = id  # Warning : paramètre immuable qui ne doit pas être modifié 
        self.version = 0 
        super().__init__(a,e,i,O,w,M0,r)  # initialise les éléments orbitaux de la particule
        
    def __eq__ (self,other) : 
        # Warning : deux particules peuvent être les mêmes mais de version differentes
        return isinstance(other, Particle) and self.id == other.id

    def __hash__ (self) : 
        # assure l'utilisation dans un set
        return hash(self.id)
    
    def UpdateOrbit (self,a,e,i,O,w,M0) : 
        self._UpdateOrbitalElements(a,e,i,O,w,M0)
        self.version += 1

    def AnomalieVraie (self,t) :
        n = 2 * pi / self.T
        M = np.mod(self.M0 + n * (t-t0) , 2 * pi)
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
# Definition de la classe ParticleSystem
"""
        
class ParticleSystem : 
    
    def __init__ (self,n,m) :
        self.particles = set()
        self.free = set(range(n,n+m))
        for i in range(n) :
            self.particles.add(Particle(i))
            
    def __iter__ (self) : return iter(self.particles)

    def __len__ (self) : return len(self.particles)

    @property
    def pairs (self) : return list(combinations(self.particles,2))

    def AddParticules (self,a,e,i,O,w,M0,r) :
        id = self.free.pop()
        self.particles.add(Particle(id,a,e,i,O,w,M0,r))
        
    def RemoveParticules (self,p) :
        self.particles.discard(p)
        self.free.add(p.id)
    
        
S = ParticleSystem(n,m-n) # Initialisation du système

"""
# Procédure de collision
"""

def BetaModel (p1,p2,E1,E2) :
    r1, r2 = p1.Position(E1), p2.Position(E2)
    v1, v2 = p1.Speed(E1), p2.Speed(E2)
    m1, m2 = p1.mass, p2.mass
    k = (r1-r2)/np.linalg.norm(r1-r2)  # Vecteur de collision
    alpha = np.dot(v1-v2, k)
    new_v1 = v1 - m2 * beta * alpha / (m1 + m2) * k
    new_v2 = v2 + m1 * beta * alpha / (m1 + m2) * k
    return new_v1,new_v2

def ComputeOrbitalElements (R,V) : 
    # Calcul des parametres orbitaux à partir des conditions initiales (cf p 114)
    # Les vecteurs sont representés en majuscules et les scalaires en minuscule
    r, v = np.linalg.norm(R), np.linalg.norm(V)
    K = array([0,0,1])                                    # vecteur e_z
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
    sin_E = sin(nu) * sqrt( abs(1 - e**2) ) / ( 1 + e * cos(nu) ) # cf p3 (valeur absolue au cas où e > 1 qui sera invalidé plus trad)
    cos_E = ( e + cos(nu) ) / ( 1 + e * cos(nu) )
    E = np.arctan2(sin_E, cos_E)                          # anomalie excentrique 
    M0 = E - e * sin_E
    return (a,e,i,Omega,omega,M0)

def ProcessCollision (p1,p2,E1,E2) :
    r1, r2 = p1.Position(E1), p2.Position(E2)
    v1, v2 = BetaModel(p1, p2, E1, E2)
    a1,e1,i1,O1,w1,M01 = ComputeOrbitalElements(r1, v1)
    p1.UpdateOrbit(a1,e1,i1,O1,w1,M01)
    a2,e2,i2,O2,w2,M02 = ComputeOrbitalElements(r2, v2)
    p2.UpdateOrbit(a2,e2,i2,O2,w2,M02)
    if not p1.CheckOrbit() :
        S.RemoveParticules(p1)
    if not p2.CheckOrbit() : 
        S.RemoveParticules(p2)



"""
# Detection de collision
"""

def DichoTaylor (p1,p2,E1,E2,t) : 

    RadiusSum = p1.radius + p2.radius
    r1, r2 = p1.Position(E1), p2.Position(E2)
    v1, v2 = p1.Speed(E1), p2.Speed(E2)
    a1, a2 = p1.Acceleration(E1), p2.Acceleration(E2)
    f = lambda h : np.linalg.norm( r1-r2 - h*(v1-v2) + 0.5*h**2*(a1-a2) ) - RadiusSum # developpement limité autour de t_min     
    
    t_inf = -delta_t   # bornes pour la dichotomie
    t_sup = 0                                                
    max_iter = 1 + int(np.log2(delta_t/tol_Taylor)) # nombre d'iterations pour obtenir la precision demandé
    
    for _ in range(max_iter) :
        t_middle = (t_sup+t_inf)/2
        if f(t_inf)*f(t_middle) < 0 :
            t_sup = t_middle
        else : 
            t_inf = t_middle
            
    return t-(t_sup+t_inf)/2
 
def Distance (p1,p2,E1,E2) :
    SumRadius = p1.radius + p2.radius
    return np.linalg.norm( p1.Position(E1) - p2.Position(E2) ) - SumRadius


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
    EList = np.linspace(0,2*np.pi,40)
    x,y,z = np.array([ p.Position(E) for E in EList ]).T
    line, = ax.plot(x, y, z, color=couleur) # permet de retourner la courbe et pas l'ensemble des points
    return line

def AffichageSphere (ax,centre,rayon,couleur,nbr_points) : 
    theta = np.linspace(0, 2 * np.pi, 2*nbr_points)
    phi = np.linspace(0, np.pi, nbr_points)
    theta, phi = np.meshgrid(theta, phi)
    x = centre[0] + rayon * np.sin(phi) * np.cos(theta)
    y = centre[1] + rayon * np.sin(phi) * np.sin(theta)
    z = centre[2] + rayon * np.cos(phi)
    return ax.plot_surface(x, y, z, color=couleur, alpha=0.8)

def AffichageVecteur (ax,point,vecteur,couleur) :
    x,y,z = point
    v_x,v_y,v_z = vecteur
    return ax.quiver(x,y,z,v_x,v_y,v_z,color=couleur)

"""
# Animation
"""

# Initialisation de l'animation    
plt.ion()
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Initialisation de la detection
E = {p.id : p.AnomalieVraie(t0) for p in S}
couleurs = {p.id : 'red' for p in S }
t = t0 + delta_t 

# Affichage de Saturne 
Saturne = AffichageSphere(ax, [0,0,0], R, 'orange',10)

# Affichage des orbites
ellipses = [ AffichageEllipse(ax,p,'blue') for p in S ]

# Affichage initiale des sphères 
spheres = [AffichageSphere(ax, p.Position(E[p.id]), p.radius, couleurs[p.id],5) for p in S]

# Affichage initiale des vecteurs vitesse 
vecteurs = [AffichageVecteur(ax, p.Position(E[p.id]), p.Speed(E[p.id]), couleurs[p.id]) for p in S]


while t < t0 + T : 

    for p in S : 
        E[p.id] = p.AnomalieVraie(t)

    # Boucle de detection
    for p1,p2 in S.pairs :
        if Distance(p1, p2, E[p1.id], E[p2.id]) <= 0 :
            CollisionCounter += 1
            t = DichoTaylor(p1, p2, E[p1.id], E[p2.id], t) 
            E[p1.id] = p1.AnomalieVraie(t)
            E[p2.id] = p2.AnomalieVraie(t) 
            ProcessCollision(p1, p2, E[p1.id], E[p2.id])
            couleurs[p1.id] = 'green'
            couleurs[p2.id] = 'green'      

        # Retrait des anciennes trajectoires et ajout des nouvelles
        if ellipses[p1.id] in ax.lines : 
            ellipses[p1.id].remove()
        if ellipses[p2.id] in ax.lines : 
            ellipses[p2.id].remove()
        if p1 in S :
            ellipses[p1.id] = AffichageEllipse(ax,p1,'blue')
        if p2 in S :
            ellipses[p2.id] = AffichageEllipse(ax,p2,'blue')        

    t += delta_t 
        
    # Retrait des sphères précédentes et ajout des nouvelles
    for sph in spheres : 
        sph.remove()
    spheres = [ AffichageSphere(ax, p.Position(E[p.id]) , p.radius, couleurs[p.id], 5) for p in S ]
        
    # Retrait des anciens vecteurs vitesses et ajout des nouveaux 
    for vect in vecteurs : 
        vect.remove()
    vecteurs = [ AffichageVecteur(ax, p.Position(E[p.id]) , p.Speed(E[p.id]), couleurs[p.id]) for p in S ]
        
    # Affichage 
    SetAxesEqual(ax)
    ax.set_title("Représentation du système")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    #ax.view_init(elev=30, azim=60)
    plt.tight_layout()
    plt.pause(0.05)


plt.ioff()
plt.show()
print(CollisionCounter)
print(len(S))


























