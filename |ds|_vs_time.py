#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 14:09:30 2026

@author: adlerbliske
"""

#import neccesary modules
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation
plt.style.use('ggplot')

#define variable parameters
m1 = m2 = r1 = r2 = 1
th1_inital = th2_initial = 1.3
#SAE is roughly accurate until 0.3 radians
#Inital angle 2*np.pi gives VERY funny results
runtime = 50
#runtime will never affect framerate or playback speed, calucations are done in advance
framerate = 10
playback_speed = 5 #default is 1
#framerate may limit speed, drop framerate for playback_speed to
#provide a real time scaling factor
dot_th1_initial = dot_th2_initial = 0
g = 9.80665 #45 latitude @ sea level

#create a function for scipy to integrate
def double_pendulum(t, state):
    th1, dot_th1, th2, dot_th2 = state

    M = ([(m1 + m2) * r1, m2 * r2 * np.cos(th1 - th2)],
         [r1 * np.cos(th1 - th2), r2])
            
    K = ([-1 * m2 * r2 * (dot_th2 ** 2) * np.sin(th1 - th2) - (m1 + m2) * g * np.sin(th1),
         r1 * (dot_th1 ** 2) * np.sin(th1 - th2) - g * np.sin(th2)])

    ddot_th1, ddot_th2 = np.linalg.solve(M, K)
    return [dot_th1, ddot_th1, dot_th2, ddot_th2]

def double_pendulum_SAE(t, state):
    th1, dot_th1, th2, dot_th2 = state

    M = [[(m1 + m2) * r1, m2 * r2],
         [r1,             r2]]
     
    K = [-(m1 + m2) * g * th1,
         -g * th2]

    ddot_th1, ddot_th2 = np.linalg.solve(M, K)
    return [dot_th1, ddot_th1, dot_th2, ddot_th2]

#integrate
initial_state = [th1_inital, dot_th1_initial, th2_initial, dot_th2_initial]
t_eval = np.linspace(0, runtime, runtime*framerate)

#method='DOP853', rtol=1e-10, atol=1e-12 is ai generated to conserve energy more effectively
solution = solve_ivp(double_pendulum, (0, runtime), initial_state, t_eval=t_eval, method='DOP853', rtol=1e-10, atol=1e-12)
solution_SAE = solve_ivp(double_pendulum_SAE, (0, runtime), initial_state, t_eval=t_eval, method='DOP853', rtol=1e-10, atol=1e-12)

#extract values
th1 = solution.y[0]
dot_th1 = solution.y[1]
th2 = solution.y[2]
dot_th2 = solution.y[3]

th1_SAE = solution_SAE.y[0]
dot_th1_SAE = solution_SAE.y[1]
th2_SAE = solution_SAE.y[2]
dot_th2_SAE = solution_SAE.y[3]

time = solution.t

#convert to cartesian
x1 = r1 * np.sin(th1)
y1 = -1 * r1 * np.cos(th1)
x2 = r1 * np.sin(th1) + r2 * np.sin(th2)
y2 = -1 * r1 * np.cos(th1) - r2 * np.cos(th2)
        
x1_SAE = r1 * np.sin(th1_SAE)
y1_SAE = -1 * r1 * np.cos(th1_SAE)
x2_SAE = r1 * np.sin(th1_SAE) + r2 * np.sin(th2_SAE)
y2_SAE = -1 * r1 * np.cos(th1_SAE) - r2 * np.cos(th2_SAE)

#remap y values to origin at (0, r1+r2)
h = r1 + r2
y1+= h
y2 += h
y1_SAE += h
y2_SAE += h

#find ds
ds_array = np.sqrt((x2_SAE - x2)**2 + (y2_SAE - y2)**2)

#find values for dp windows
x_min = np.min([x2, x2_SAE])
x_max = np.max([x2, x2_SAE])
y_min = np.min([y2, y2_SAE])
y_max = np.max([y2, y2_SAE])
ds_max = np.max(ds_array)

#define mosaic
plot, ax = plt.subplot_mosaic([['A', 'B', 'B', 'B']], layout = 'constrained')

#define plots
dp, = ax['A'].plot(0, 0, marker='o', markerfacecolor='r', color='k')
dp_SAE, = ax['A'].plot(0, 0, marker='o', markerfacecolor='b', color='k')
path, = ax['A'].plot(0,0, color='r', alpha=0.2)
path_SAE, = ax['A'].plot(0,0, color='b', alpha=0.2)
ds_line, = ax['A'].plot(0,0,color='purple', alpha=0.7)

x2_v_t, = ax['B'].plot(0,0, color='r', alpha=0.75)
x2_SAE_v_t, = ax['B'].plot(0,0, color='b', alpha=0.5)

ds, = ax['B'].plot(0,0,color='purple', alpha=0.7)

#set window
ax['A'].set_aspect('equal')
ax['A'].set_xlim(x_min - 0.1, x_max + 0.1)
if y_max < r1 + r2: ax['A'].set_ylim(y_min - 0.1 , (r1 + r2) + 0.1)
else: ax['A'].set_ylim(y_min - 0.1, y_max + 0.1)

ax['A'].set_title('Real vs SAE')

ax['B'].set_xlim(0,runtime)
ax['B'].set_ylim(0,ds_max)

ax['B'].set_ylabel('|ds| in meters {m}')
ax['B'].set_xlabel('Time in seconds {s}')

#animate
def animate(i):
    dp.set_xdata([0, x1[i], x2[i]])
    dp.set_ydata([h, y1[i], y2[i]])

    dp_SAE.set_xdata([0, x1_SAE[i], x2_SAE[i]])
    dp_SAE.set_ydata([h, y1_SAE[i], y2_SAE[i]])
  
    ds_line.set_xdata([x2[i], x2_SAE[i]])
    ds_line.set_ydata([y2[i], y2_SAE[i]])
    
    ds.set_xdata(time[:i])
    ds.set_ydata(ds_array[:i])

    return dp, dp_SAE, ds, ds_line

anim = FuncAnimation(plot, animate, 
                     frames=runtime*framerate, 
                     interval=1000/framerate/playback_speed, 
                     blit=True)
plt.show()