#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Iterator
import numpy as np 
from numpy import pi,cos,sin,sqrt,array
from numpy.linalg import norm
import matplotlib
matplotlib.use("macosx")
import random as rd
import matplotlib.pyplot as plt
import heapq as hq
from itertools import combinations # pour pouvoir generer les couples non ordonnés d'un set 
from functools import total_ordering

from time import time
import logging

logging.basicConfig(
    level=logging.INFO,  # Niveau de base (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(levelname)s - %(message)s',  # Format du message
    filename = 'simulation.log',  # Fichier de sortie (ou None pour console)
    filemode='w'  # 'w' pour écraser à chaque exécution, 'a' pour ajouter
)

import csv
import os

EXPORT_FILENAME = "data/donnees_sim.csv"

# Supprimer le fichier existant au démarrage
if os.path.exists(EXPORT_FILENAME):
    os.remove(EXPORT_FILENAME)

# Écrire l'entête une seule fois
with open(EXPORT_FILENAME, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["t", "id", "a", "e", "i", "O", "w", "m"])  # a: demi-grand axe, e: excentricité, etc.

"""
# PARAMETRES DU SYSTEME #
"""

G = 1        # Constante de gravitation
M = 1        # Masse de l'astre (masse unitaire)
R = 0.355    # Rayon de l'astre 
mu = G * M

n = 100      # Nombre de particules
rho = 7.12   # masse volumique de chaque particule du système à l'échelle 

R1 = 1       # Periastre minimal de la distribution (longueur unitaire defini comme égale à la limite de Roche)
R2 = 3       # Apoastre maximal de la distribution

init_radius = 0.023  * R
init_mass = 4 * pi * init_radius**3 * rho / 3
M_tot = n * init_mass

beta = 1.4   # paramètre d'elasticité des chocs (compris dans [1,2])

global_t = 0             # instant initial
T = 1500                  # Periode de cycle long
NbrCycleLong = 100         # Nombre maximal de cycle long effectué par le système 

CollisionCounter, NbrFusion, NbrCrash, NbrEjection, next_export = 0, 0, 0, 0, 0 
MaxCollision = np.inf
dN_export = 10    # Intervalle (en nombres de collisions) entre chaque snapshot dy système

eps = 0.01  # ordre de perturbations des orbites lors du clonage de particule

tol_NR = 1e-8   # precision demandé pour résoudre l'equation de Kepler 
max_iter = 10   # nombre d'iterations maximal pour Newton-Raphson

tol_Taylor = 1e-8  # precision demandé sur la distance lors du calcul de l'instant de collision

max_loop_number = 100  # nombre maximum de boucle avant de lever une erreur sur la generation de particules valides

max_execution_time = 10 * 3600 # temps maximal d'execution

"""
# Resolution numerique de l'equation de Kepler
"""

solve_kepler_time = 0

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
    global solve_kepler_time
    t0 = time()
    M = np.mod(M,2*pi)
    E,convergence = KeplerNewtonRaphson(M, e)
    if not convergence : 
        E = KeplerDichotomie(M, e)
    t1 = time()
    solve_kepler_time += t1-t0
    return E
    
    


"""
# Fonctions auxiliaires de selection aleatoire des paramètres orbitaux 
"""

def RandomConicParameters () : 
    global R1, R2
    # Selection aléatoire des paramètres de l'ellipse
    # selection tel que la trajectoire soit comprise entre les sphères de rayon R1 et R2
    a = rd.uniform(R1,R2)                # demi-grand axe a
    e = rd.uniform(0,0.99)              # excentricité e
    while a*(1-e) < R1 or a*(1+e) > R2 : 
        a = rd.uniform(R1,R2)            
        e = rd.uniform(0,0.999)
    return a,e

def RandomEulerAngles () : 
    # Selection aléatoire des angles d'Euler paramètrant la trajectoire en 3D
    # l'inclinaison est choisie tel que cos(i) soit uniforme dans [-1,1]
    i = np.arccos(rd.uniform(-0.3,0.3))   # inclinaison i
    O = rd.uniform(0,2*pi)            # longitude du noeud ascendant O
    w = rd.uniform(0,2*pi)            # argument du periastre w
    return i,O,w

"""
# Definition de la classe OrbitalElements
"""

class OrbitalElements :
    
    def __init__ (self,a=None,e=None,i=None,O=None,w=None,E0=None,mass=None) :
        global init_mass
        if not a or not e or not i or not O or not w or not E0 : 
            a,e = RandomConicParameters()
            i,O,w = RandomEulerAngles()
            E0 = rd.uniform(0,2*pi)  # Anomalie excentrique initiale  
            mass = init_mass
        self.a,self.e,self.i,self.O,self.w,self.E0,self.mass = a,e,i,O,w,E0,mass
        self._ComputeOrbitParameters()
        self._ComputeOrbitRotationalMatrix()

    @property        
    def Periastre (self) : return self.a * ( 1 - self.e ) 
    @property
    def Apoastre (self) : return self.a * ( 1 + self.e )
        
    def _ComputeOrbitParameters (self) :
        global rho
        self.M0 = self.E0 - self.e * sin( self.E0 ) # Anomalie moyenne initiale M0 
        self.radius = ( 3 * self.mass / ( 4 * pi * rho ) )**(1/3)  # rayon
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
        
    def _UpdateOrbitalElements (self,a,e,i,O,w,M0) : 
        m = 0.99 * self.mass    # modélise des pertes de 1% de la masse lors des collisions
        self.a,self.e,self.i,self.O,self.w,self.M0,self.mass = a,e,i,O,w,M0,m
        self._ComputeOrbitParameters()
        self._ComputeOrbitRotationalMatrix()

    def _GeneratePerturbedOrbitalElements (self) : 
        global eps
        a = self.a * ( 1 + rd.uniform(-eps,eps) )
        e = self.e * ( 1 + rd.uniform(-eps,eps) )
        i = self.i * ( 1 + rd.uniform(-eps,eps) )
        O = self.O * ( 1 + rd.uniform(-eps,eps) )
        w = self.w * ( 1 + rd.uniform(-eps,eps) )
        E0 = rd.uniform(0,2*pi)
        return a, e, i, O, w, E0, self.mass
    
    def CheckOrbit (self) : 
        if self.e >= 0.99 or self.Periastre + self.radius < R : return False 
        else : return True

    def AdjustMass (self,m) : 
        self.mass = m 
        self.radius = ( 3 * self.mass / ( 4 * pi * rho ) )**(1/3)

    
            
"""
# Definition de la classe Particle héritant de la classe OrbitalElements
"""  
    
@total_ordering
class Particle (OrbitalElements) :
    
    def __init__ (self,id,a=None,e=None,i=None,O=None,w=None,E0=None,r=None) :
        self.id = id  # Warning : paramètre immuable qui ne doit pas être modifié 
        self.version = 0 
        self.t_collision = global_t
        super().__init__(a,e,i,O,w,E0,r)  # initialise les éléments orbitaux de la particule
        
    def __eq__ (self,other) : 
        # Warning : deux particules peuvent être les mêmes mais de version differentes
        return isinstance(other, Particle) and self.id == other.id
    
    def __lt__ (self, other) :
        return isinstance(other, Particle) and self.id < other.id

    def __hash__ (self) : 
        # assure l'utilisation dans un set
        return hash(self.id)
    
    def UpdateOrbit (self,a,e,i,O,w,M0,t) : 
        self._UpdateOrbitalElements(a,e,i,O,w,M0)
        self.version += 1
        self.t_collision = t

    def GeneratePerturbedClone (self, id) :
        a, e, i, O, w, E0, m = self._GeneratePerturbedOrbitalElements()
        return Particle(id, a, e, i, O, w, E0, m)

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
# Procédure de collision
"""

def Distance (p1 : Particle, p2 : Particle, E1, E2) :
    SumRadius = p1.radius + p2.radius
    return np.linalg.norm( p1.Position(E1) - p2.Position(E2) ) - SumRadius


def CheckRelativeSpeed (p1 : Particle, p2 : Particle, E1, E2) :
    v1, v2 = p1.Speed(E1), p2.Speed(E2)
    return norm(v1-v2) > 0.1 * min(norm(v1), norm(v2))


def BetaModel (p1 : Particle,p2 : Particle, E1, E2, beta=beta) :
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
    
    @property
    def total_mass (self) : return sum([p.mass for p in self])
    
    def Export (self) : 
        with open(EXPORT_FILENAME, mode='a', newline='') as f:
            writer = csv.writer(f)
            for p in S :
                id, a, e, i, O, w, m = p.id, p.a, p.e, p.i, p.O, p.w, p.mass
                writer.writerow([global_t, id, a, e, i, O, w, m])
    
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
        loop_counter = 0
        while not self._ValidParticle(new_p) : 
            new_p = Particle(id)
            loop_counter += 1
            if loop_counter >= max_loop_number : 
                raise KeyError ("Pas assez de place dans le système")
        self.particles.add(new_p)

    def _AddPerturbedClone (self) : 
        # Ajoute au système un clone valide d'une particule aléatoire du système 
        p : Particle = rd.choice(list(self.particles))
        id = self._NewIndex()
        new_p = p.GeneratePerturbedClone(id)
        loop_counter = 0
        while not self._ValidParticle(new_p) : 
            p : Particle = rd.choice(list(self.particles))
            new_p = p.GeneratePerturbedClone(id)
            loop_counter += 1
            if loop_counter >= max_loop_number : 
                raise KeyError ("Pas assez de place dans le système")
        self.particles.add(new_p)
        logging.info("Ajout de la particule %d cloné à partir de %d", new_p.id,p.id )
        return new_p  # necessaire pour l'affichage de l'animation

    def _UpdateParticle (self, p : Particle, r, v, t) :
        global NbrCrash, NbrEjection
        # Met à jour une particule après une collision et la remplace si éjecté 
        a, e, i, O, w, E0 = ComputeOrbitalElements(r, v)
        p.UpdateOrbit(a, e, i, O, w, E0, t)
        new_p = None 
        if p.Ejected() :
            NbrEjection += 1
            self._RemoveParticle(p)
            new_p = self._AddPerturbedClone()
        elif p.Crashed() : 
            NbrCrash += 1
            self._RemoveParticle(p)
            new_p = self._AddPerturbedClone()
        return new_p
    
    def _ProcessStandardCollision (self, p1 : Particle, p2 : Particle, t) :
        E1, E2 = p1.AnomalieExcentrique(t), p2.AnomalieExcentrique(t) 
        r1, r2 = p1.Position(E1), p2.Position(E2)
        v1, v2 = BetaModel(p1, p2, E1, E2)
        new_p1 = self._UpdateParticle (p1, r1, v1, t)
        new_p2 = self._UpdateParticle (p2, r2, v2, t)
        return [new_p1, new_p2]
    
    def _ProcessFusion (self, p1 : Particle, p2 : Particle, t) :
        global rho, NbrFusion
        logging.info("Fusion des particules %d et %d", p1.id, p2.id)
        self._RemoveParticle(p1) 
        self._RemoveParticle(p2) 
        E1, E2 = p1.AnomalieExcentrique(t), p2.AnomalieExcentrique(t) 
        m1, m2 = p1.mass, p2.mass
        new_m = 0.999 * (m1 + m2)  # modelise des pertes de masses de 0.1 % pour une collision faiblement dissipative
        r = (m1 * p1.Position(E1) + m2 * p2.Position(E2))/(m1 + m2)
        v = (m1 * p1.Speed(E1) + m2 * p2.Speed(E2))/(m1 + m2)
        a, e, i, O, w, E0 = ComputeOrbitalElements(r, v)
        id = self._NewIndex()
        new_p = Particle(id,a,e,i,O,w,E0,new_m)
        self.particles.add(new_p)
        NbrFusion += 1
        cloned_p = self._AddPerturbedClone()
        return [new_p, cloned_p]

    def ProcessCollision (self, p1 : Particle, p2 : Particle, t) :
        global CollisionCounter 
        processed_particules : list[Particle] = []
        E1, E2 = p1.AnomalieExcentrique(t), p2.AnomalieExcentrique(t) 
        if norm(p1.Speed(E1)-p2.Speed(E2)) < 0.2 :  # valeur experimentale
            processed_particules = self._ProcessFusion(p1,p2,t)
        else : 
            processed_particules = self._ProcessStandardCollision(p1, p2, t)
        CollisionCounter += 1
        return processed_particules
    
    def RecalibrateSystem (self) : 
        global M_tot
        total_mass = S.total_mass
        logging.info("Système recalibré d'un facteur %s",  M_tot / total_mass)
        for p in self : 
            new_mass = p.mass * M_tot / total_mass
            p.AdjustMass(new_mass)



"""
# Detection de collision
"""

def CheckFiltreSpatial (p1 : Particle, p2 : Particle) :
    RadiusSum = p1.radius + p2.radius
    return not  (p1.Periastre - p2.Apoastre > RadiusSum or p2.Periastre - p1.Apoastre > RadiusSum)


def CrudeDicho (p1 : Particle,p2 : Particle, t, dt) : 
    # 5 fois plus lente que sur la distance approché mais s'effectue en ~1ms pour ~20 boucles
    # Initialisation des bornes et redefintion de la fonction distance pour concision
    Dist = lambda t : Distance (p1, p2, p1.AnomalieExcentrique(t), p2.AnomalieExcentrique(t))
    t_inf, t_middle, t_sup = t-dt, t-dt/2, t

    # Si l'intervalle n'est pas valide, on renvoit t 
    if Dist(t_inf) < 0 : 
        logging.warning("Dichotomie non valide")
        return t 

    # Recherche dichotomique :
    else : 
        while abs(Dist(t_middle)) > tol_Taylor :
            if Dist(t_inf) * Dist(t_middle) < 0 : 
                t_sup = t_middle
            else : 
                t_inf = t_middle
            t_middle = (t_sup+t_inf)/2

        return t_middle
    

def ComputeAdaptativeStep (p1 : Particle, p2 : Particle, E1, E2, Dist, local_tol=0.5, orbital_tol=0.01) :
    # Initialisation
    R1, R2 = p1.radius, p2.radius
    v1, v2 = norm(p1.Speed(E1)), norm(p2.Speed(E2))
    SumRadius = R1 + R2

    # Calcul des bornes de delta_t : 
    dt_min = 0.2 * min(R1 / v1, R2 / v2)
    dt_max = 0.1 * min(p1.P / v1, p2.P / v2)

    # Calcul du pas à la bonne echelle (locale ou orbitale) :
    if Dist < 100 * SumRadius : 
        local_dt = local_tol * min(R1 / v1, R2 / v2) * (Dist / SumRadius)
        return np.clip(local_dt, dt_min, dt_max)
    else : 
        AvgDist = abs(p1.a -p2.a) + abs(p1.e - p2.e) * (p1.a + p2.a)/2
        orbital_dt = orbital_tol * min(p1.P / v1, p2.P / v2) * (Dist / AvgDist)
        return np.clip(orbital_dt, dt_min, dt_max)



def DetectionCollision (p1 : Particle, p2 : Particle) : 
    #logging.info("Recherche de collision entre %d et %d", p1.id, p2.id)
    # Initialisation
    Collision = False
    E1, E2 = p1.AnomalieExcentrique(global_t), p2.AnomalieExcentrique(global_t)
    Dist = Distance(p1, p2, E1, E2)

    # Initialisation du temps (ajout d'un pas de temps car aucune particule est en collision à global_t)
    dt = ComputeAdaptativeStep(p1, p2, E1, E2, Dist )
    t = global_t + dt

    # Boucle de detection
    if CheckFiltreSpatial(p1, p2) :
        while t < global_t + T and not Collision : 
            Dist = Distance(p1, p2, p1.AnomalieExcentrique(t), p2.AnomalieExcentrique(t))
            if Dist <= 0 : 
                Collision = True 
                t = CrudeDicho(p1, p2, t, dt)
            else : 
                dt = ComputeAdaptativeStep(p1, p2, E1, E2, Dist)
                t += dt
    
    return t, Collision


"""
# Evolution du système
"""
t0 = time()
S = ParticleSystem(n) # Initialisation du système
t1 = time()

LongCycleInitTime = 0 
ShortCycleTime = 0

logging.info("Système initialisé")

def AppendCollision (p1 : Particle, p2 : Particle, CollisionQueue ) :
    t,Collision = DetectionCollision(p1, p2)
    if Collision :
        hq.heappush(CollisionQueue, (t, p1.version, p2.version, p1, p2))


def CheckValidCollision (p1 : Particle, p2 : Particle, p1_version, p2_version) : 
    if p1 in S and p2 in S and p1.version == p1_version and p2.version == p2_version : 
        return True 
    else : 
        return False

        
def ShortCycle (CollisionQueue : list[tuple[float,int,int,Particle,Particle,]]) : 
    global global_t, next_export
    t, p1_version, p2_version, p1, p2 = hq.heappop(CollisionQueue)
    collision = False
    if CheckValidCollision(p1, p2, p1_version, p2_version) :
        global_t = t
        collision = True
        processed_particules = S.ProcessCollision (p1, p2, t)
        while processed_particules != [] : 
            p0 = processed_particules.pop()  # attention p0 peut etre de type None
            if p0 : 
                for p in S : 
                    if p not in processed_particules and p != p0 :
                        AppendCollision(p0, p, CollisionQueue)
    if CollisionCounter >= next_export : 
        S.Export()
        logging.info("Système exporté")
        next_export += dN_export
    return collision


def LongCycle (k) :
    global global_t, LongCycleInitTime, ShortCycleTime, next_export, dN_export
    logging.error("Début du %d-ieme cycle long à t = %s", k, global_t)
    logging.info("Nombre de collisions : %d", CollisionCounter)
    S.Export()
    next_export += dN_export
    logging.info("Système exporté")
    S.RecalibrateSystem()  # recalibre la masse totale du système pour garder une masse constante 
    HappenedCollision = False
    CollisionQueue : list[tuple[float,int,int,Particle,Particle,]] = []
    t0 = time()
    for p1,p2 in S.pairs :
        AppendCollision(p1, p2, CollisionQueue)
    t1 = time()
    logging.info("Fin de l'initialisation du cycle long")
    while CollisionQueue != [] : 
        happened_collision = ShortCycle(CollisionQueue)
        HappenedCollision = HappenedCollision or happened_collision
    if not HappenedCollision : 
        global_t += T
    t2 = time()
    LongCycleInitTime += t1-t0
    ShortCycleTime += t2-t1

                    
def Evolution () : 
    global max_execution_time
    t_0 = time()
    for k in range(NbrCycleLong) :
        t_1 = time()
        t_tot = t_1-t_0
        if CollisionCounter >= MaxCollision or t_tot >= max_execution_time : 
            break 
        LongCycle(k)     


"""
# Mise en marche de l'algorihtme 
"""

if __name__ == "__main__":
    t2 = time()
    Evolution()
    S.Export()
    logging.info("Système exporté")
    t3 = time()

    t_tot = t3-t0

    logging.info("Nombre de collision : %d", CollisionCounter)
    logging.info("Nombre de fusion : %d", NbrFusion)
    logging.info("Nombre de crash : %d", NbrCrash)
    logging.info("Nombre d'ejection : %d", NbrEjection)
    logging.info("Nombre de particule initial : %d et final %d", n, len(S))
    logging.info("Temps d'initialisation : %s sec", t1-t0)
    logging.info("Temps d'initialisation des cycles longs : %s pourcent du temps total", LongCycleInitTime/t_tot * 100)
    logging.info("Temps des cycles courts : %s pourcent du temps total", ShortCycleTime/t_tot * 100)
    logging.info("Temps de resolution de Kepler : %s pourcent du temps total", solve_kepler_time/t_tot*100)
    logging.info("Temps total de simulation : %s min", t_tot/60)





        
        



            

    
        
            











































        
    



















