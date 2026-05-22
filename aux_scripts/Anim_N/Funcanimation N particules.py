#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Iterator
import numpy as np 
from numpy import pi,cos,sin,sqrt,array
import matplotlib
matplotlib.use("macosx")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D as ax
from matplotlib.animation import FuncAnimation, FFMpegWriter
import random as rd
from itertools import combinations # pour pouvoir generer les couples non ordonnés d'un set 
from time import time

import logging
logging.basicConfig(
    level=logging.INFO,  # Niveau de base (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(levelname)s - %(message)s',  # Format du message
    filename = 'animation.log',  # Fichier de sortie (ou None pour console)
    filemode='w'  # 'w' pour écraser à chaque exécution, 'a' pour ajouter
)

"""
# PARAMETRES DU SYSTEME #
"""

G = 1        # Constante de gravitation
M = 1        # Masse de l'astre (masse unitaire)
R = 0.355    # Rayon de l'astre 
mu = G * M

n = 50      # Nombre de particules
rho = 7.12   # masse volumique de chaque particule du système à l'échelle 

R1 = 1       # Periastre minimal de la distribution (longueur unitaire defini comme égale à la limite de Roche)
R2 = 2       # Apoastre maximal de la distribution

radius = R/10

beta = 1.5   # paramètre d'elasticité des chocs (compris dans [1,2])

global_t = 0                    # instant initial
pas_discretisation = 50  # pas de discrétisation du temps
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
        self.C = sqrt( self.p * mu )                # Constante des aires (ou moment angulaire spécifique)
        
    def _ComputeOrbitRotationalMatrix (self) :
        cos_i,sin_i = cos(self.i),sin(self.i)   
        cos_O,sin_O = cos(self.O),sin(self.O)
        cos_w,sin_w = cos(self.w),sin(self.w)
        self.R = array([[ cos_O*cos_w - sin_O*sin_w*cos_i , - cos_O*sin_w - sin_O*cos_w*cos_i , sin_O*sin_i ],
                          [ sin_O*cos_w + cos_O*sin_w*cos_i , - sin_O*sin_w + cos_O*cos_w*cos_i , - cos_O*sin_i ],
                          [ sin_w*sin_i , cos_w*sin_i , cos_i ]])
        
    def _UpdateOrbitalElements (self,a,e,i,O,w,E0) : 
        self.a,self.e,self.i,self.O,self.w,self.E0 = a,e,i,O,w,E0
        self._ComputeOrbitParameters
        self._ComputeOrbitRotationalMatrix()

    @property        
    def Periastre (self) : return self.a * ( 1 - self.e ) 
    @property
    def Apoastre (self) : return self.a * ( 1 + self.e )
    
    def CheckOrbit (self) : 
        if self.e >= 1 or self.Periastre + self.radius < R : return False 
        else : return True

    def _GeneratePerturbedOrbitalElements (self) : 
        a = self.a * ( 1 + rd.uniform(-0.01,0.01) )
        e = self.e * ( 1 + rd.uniform(-0.01,0.01) )
        i = self.i * ( 1 + rd.uniform(-0.01,0.01) )
        O = self.O * ( 1 + rd.uniform(-0.01,0.01) )
        w = self.w * ( 1 + rd.uniform(-0.01,0.01) )
        E0 = rd.uniform(0,2*pi)
        return a, e, i, O, w, E0, self.radius

            

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
    
    def UpdateOrbit (self,a,e,i,O,w,E0,t) : 
        self._UpdateOrbitalElements(a,e,i,O,w,E0)
        self.version += 1
        self.t_collision = t

    def GeneratePerturbedClone (self, id) :
        a, e, i, O, w, E0, r = self._GeneratePerturbedOrbitalElements()
        return Particle(id, a, e, i, O, w, E0, r)

    def Ejected (self) : 
        if self.e >= 1 : 
            logging.info("Particule %d s'est fait ejecté du système", self.id)
            return True
        else : return False 

    def Crashed (self) : 
        if self.Periastre <= R + self.radius : 
            logging.info("Particule %d s'est crashé dans l'astre", self.id)
            return True
        else : return False

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
# Modèle de collision et calcul des nouveaux paramètres orbitaux
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


def ComputeOrbitalElements (R,V) : 
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

"""
# Fonction de distances
"""

def Distance (p1 : Particle,p2 : Particle, E1, E2) :
    SumRadius = p1.radius + p2.radius
    return np.linalg.norm( p1.Position(E1) - p2.Position(E2) ) - SumRadius

def DistanceAstre (p : Particle, E) : 
    return p.OrbitalRadius(E) - p.radius - R

"""
# Definition de la classe ParticleSystem
"""
        
class ParticleSystem : 
    
    def __init__ (self,n) :
        self.particles = set()
        self.next_index = 0
        for _ in range(n) :
            self._AddRandomParticle()
            
    def __iter__ (self) -> Iterator[Particle] :
        return iter(self.particles)

    def __len__ (self) : return len(self.particles)

    def __repr__ (self) : return repr([p.id for p in S])

    @property
    def pairs (self) -> list[tuple[Particle,Particle]] :
        return list(combinations(self.particles,2))
    
    def _RemoveParticle (self, p : Particle) : self.particles.discard(p)

    def _NewIndex (self) : 
        id = self.next_index
        self.next_index += 1 
        return id 
    
    def _ValidParticle (self, new_p : Particle) :
        # check si aucune particule ne penetre la nouvelle particule 
        # prérequis : la nouvelle particule est crée à l'instant global_t
        for p in self : 
            E = p.AnomalieExcentrique(global_t)
            if Distance(new_p, p, new_p.E0, E) <= 0 :
                return False
        return True
    
    def _AddRandomParticle (self) : 
        # Ajoute au système une nouvelle particule aléatoire valide
        id = self._NewIndex()
        new_p = Particle(id)
        while not self._ValidParticle(new_p) : 
            new_p = Particle(id)
        self.particles.add(new_p)
    
    def _AddPerturbedClone (self) : 
        # Ajoute au système un clone valide d'une particule aléatoire du système 
        p : Particle = rd.choice(list(self.particles))
        id = self._NewIndex()
        new_p = p.GeneratePerturbedClone(id)
        while not self._ValidParticle(new_p) : 
            p : Particle = rd.choice(list(self.particles))
            new_p = p.GeneratePerturbedClone(id)
        self.particles.add(new_p)
        logging.info("Ajout de la particule %d cloné à partir de %d", new_p.id,p.id )
        return new_p  # necessaire pour l'affichage de l'animation

    def _UpdateParticle (self, p : Particle, r, v, t) :
        # Met à jour une particule après une collision et la remplace si éjecté 
        a, e, i, O, w, E0 = ComputeOrbitalElements(r, v)
        p.UpdateOrbit(a, e, i, O, w, E0, t)
        new_p = None 
        if p.Ejected() or p.Crashed() : 
            self._RemoveParticle(p)
            new_p = self._AddPerturbedClone()
        return new_p

    def ProcessCollision (self, p1 : Particle, p2 : Particle, E1, E2, t) :
        r1, r2 = p1.Position(E1), p2.Position(E2)
        v1, v2 = BetaModel(p1, p2, E1, E2)
        new_p1 = self._UpdateParticle (p1, r1, v1, t)
        new_p2 = self._UpdateParticle (p2, r2, v2, t)
        return new_p1, new_p2

t0 = time()
S = ParticleSystem(n) # Initialisation du système
t1 = time()


"""
# Detection de collision
"""

def DichoTaylor (p1 : Particle,p2 : Particle, E1, E2, t) : 

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

def AffichageEllipse (ax,p : Particle,couleur) :
    # line, permet de retourner la courbe et pas l'ensemble des points
    EList = np.linspace(0,2*np.pi,40)
    x,y,z = np.array([ p.Position(E) for E in EList ]).T
    line, = ax.plot(x, y, z, color=couleur, linewidth = 0.75) 
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
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Initialisation de la detection
Objects3D = {}


def init () :
    global Objects3D
    Saturne = AffichageSphere(ax, [0,0,0], R, 'orange',10)
    obj = [Saturne]
    for p in S : 
        E, couleur = p.AnomalieExcentrique(global_t), 'red'
        r, v = p.Position(E), p.Speed(E)
        Objects3D[p] = array([E, couleur,  AffichageEllipse(ax,p,'blue'), AffichageSphere(ax, r, p.radius, couleur,10), AffichageVecteur(ax, r, v, couleur) ])
        obj += [Objects3D[p][2], Objects3D[p][3], Objects3D[p][4]]
    return obj


def Update3Dparticule (p : Particle, collision) :
    global Objects3D
    new_obj = []

    if p not in Objects3D.keys() :
        logging.critical("particule %d absent de Objects3D ",p.id)
    Objects3D[p][3].remove()
    Objects3D[p][4].remove()

    if collision : 

        Objects3D[p][2].remove()
        Objects3D[p][1] = 'green'

        if p in S : 
            E, couleur = Objects3D[p][0], Objects3D[p][1]
            r, v = p.Position(E), p.Speed(E)
            Objects3D[p][2] = AffichageEllipse(ax,p,'blue')
            Objects3D[p][3] = AffichageSphere(ax, r, p.radius, couleur,10)
            Objects3D[p][4] = AffichageVecteur(ax, r, v, couleur)
            new_obj = [Objects3D[p][2], Objects3D[p][3], Objects3D[p][4]]

        else : 
            del Objects3D[p]
    
    else : 
        E, couleur = Objects3D[p][0], Objects3D[p][1]
        r, v = p.Position(E), p.Speed(E)
        Objects3D[p][3] = AffichageSphere(ax, r, p.radius, couleur,10)
        Objects3D[p][4] = AffichageVecteur(ax, r, v, couleur)
        new_obj = [Objects3D[p][3], Objects3D[p][4]]

    return new_obj


def update (frame) :
    global global_t, Objects3D, CollisionCounter
    new_obj = []
    CollisionPairsList : list[tuple[Particle,Particle]] = []
    Collisions = {p : False for p in S}
    NewParticlesList : list[Particle] = []

    # Actualisation des positions
    for p in S : 
        Objects3D[p][0] = p.AnomalieExcentrique(global_t)
    
    # Boucle de detection
    for p1,p2 in S.pairs :
        if Distance(p1, p2, Objects3D[p1][0], Objects3D[p2][0]) <= 0 :
            CollisionCounter += 1
            Collisions[p1] = True
            Collisions[p2] = True
            CollisionPairsList.append((p1,p2))

    # Actualisation de toutes les collisions dans le système 
    for p1, p2 in CollisionPairsList : 
            t_coll = DichoTaylor(p1, p2, Objects3D[p1][0], Objects3D[p2][0], global_t)
            E1, E2 =  p1.AnomalieExcentrique(t_coll), p2.AnomalieExcentrique(t_coll)
            logging.info("Collision détectée entre %d et %d à t=%s", p1.id, p2.id, t_coll)
            new_p1, new_p2 = S.ProcessCollision(p1, p2, E1, E2, t_coll)
            NewParticlesList.append(new_p1)
            NewParticlesList.append(new_p2)
            Objects3D[p1][0] = p1.AnomalieExcentrique(global_t)
            Objects3D[p2][0] = p2.AnomalieExcentrique(global_t)

    # Actualisation des objets 3D
    for p in Collisions.keys() :
        new_obj += Update3Dparticule(p,Collisions[p])

    # Affichage des nouvelles particules générés 
    for p in NewParticlesList : 
        if p : 
            E, couleur = p.AnomalieExcentrique(global_t), 'cyan'
            r, v = p.Position(E), p.Speed(E)
            Objects3D[p] = array([E, couleur,  AffichageEllipse(ax,p,'blue'), AffichageSphere(ax, r, p.radius, couleur,10), AffichageVecteur(ax, r, v, couleur) ])
            new_obj += [Objects3D[p][2], Objects3D[p][3], Objects3D[p][4]]

    global_t += delta_t

    # Affichage 
    SetAxesEqual(ax)
    ax.set_title("Représentation du système")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    return new_obj

t2 = time()
ani = FuncAnimation(fig, update, init_func=init, frames=500, interval=60, blit=False, repeat=False)


logging.info("Système initial :" + repr(S))

#"""
writer = FFMpegWriter(fps=30, bitrate=10000)
ani.save("animation.mp4", writer=writer)
#"""

#plt.show()
t3 = time()
logging.info("Système final :" + repr(S) + " de taille %d",len(S))
logging.info("Durée d'initialisation : %s",t1-t0)
logging.info("Durée de génération de l'animation : %s", (t3-t2)/60)



















