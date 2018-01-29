# Generate a template for detector strain
# as a function of frequency using eq. 3.4
# in Allen et. al. 2011 (title: FINDCHIRP)
# arXiv:gr-qc/0509116
# 
# Adopted from a module
# by Ashley Disbrow 2013


import cmath
import numpy as np
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

###########################################
#Define all functions which generate template
###########################################

#Calculate strain using frequency domain
def h(f, Amp, Psi, D_eff,imaginary):
    h_f = (1.0/D_eff)*Amp*f**(-7./6)*np.exp(-imaginary*Psi)
    return h_f

#Amplitude equation 3.4b
def Amp(ang_mom):
    G=6.67e-11 #Units: m^3/(kg*s^2)
    c=3e8 #m/s
    Mpc = 3e22 #m
    m_sun = 2e30 #kg
    c1=-math.sqrt(5./(24*math.pi)) #first term eq. 3.4b
    c2=(G*m_sun/(c**2*Mpc)) #unitless
    c3=(math.pi*G*m_sun/c**3)**(-1./6) #(T)**-(1/6)
    c4=ang_mom**(5./6) #unitless
    return c1*c2*c3*c4 #with units of (time)**-(1/6)
    
#eq. 3.4c
def Psi(f,eta,Mtot):
    G=6.67e-11*2e30 #Units: m^3/(M_sun*s^2)
    c=3e8
    v = (G*Mtot*math.pi*f/c**3)**(1./3) #unitless
    t1=(3715./756+55.*eta/9)
    t2=15293365./508032+27145.*eta/504+3085.*eta**2/72
    t_0 = 0 #assume time is negative until coalescence
    phi_0 = 0 #equal to Phi_c, assuming i=1 and F_cross=0
    Psi = 2*math.pi*f*t_0-2*phi_0-math.pi/4+(3./(128*eta))*(v**(-5)+t1*v**(-3)-16*math.pi*v**(-2)+t2/v)
    return Psi
                                                      

##############################################
#Main - Give parameters and generate template
##############################################

def createTemplate(fs,dataChunk, m1, m2):
    
    #Definitions necessary for math
    Deff = 1. #Mpc
    j = cmath.sqrt(-1) #define the imaginary variable

    dt=dataChunk/2
    Mtot=m1+m2
    eta = float(m1*m2)/(Mtot**2)
    ang_mom=eta**(3./5)*Mtot

    # Create array of frequencies
    nyquist = fs/2
    f_i = 1./(2*dt)
    # f_i = 25
    frequency = np.arange(0,nyquist+1./(2*dt),1./(2*dt))
    frequency[0] = 1./(2*dt)
    

    #use frequencies to find strain:
    amplitude = Amp(ang_mom)
    psi_vector = Psi(frequency,eta,Mtot)
    strain = h(frequency,amplitude,psi_vector,Deff,j)

    # The template should stop at f_isco, the innermost
    # stable circular orbit.
    c = 3e8
    G = 6.67e-11
    m_sun = 2e30 #kg
    f_isco = c**3/(6*math.sqrt(6)*math.pi*G*m_sun*Mtot)
    strain[f_isco<frequency]=0

    return strain,frequency
