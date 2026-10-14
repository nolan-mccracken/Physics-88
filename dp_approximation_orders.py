#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 21:11:46 2026

@author: adlerbliske
"""

#import modules
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation
plt.style.use('ggplot')

#define variable parameters
m1 = m2 = r1 = r2 = 1
th1_inital = th2_initial = 1.6
#O(th) is roughly accurate until 0.3 radians
#O(th**3) breaks the program past 1.3 radians
dot_th1_initial = dot_th2_initial = 0
runtime = 100
framerate = 60
g = 9.80665 #45 latitude @ sea level on Earth

def double_pendulum(t, state):
    th1, dot_th1, th2, dot_th2 = state

    M = ([(m1 + m2) * r1, m2 * r2 * np.cos(th1 - th2)],
         [r1 * np.cos(th1 - th2), r2])
            
    K = ([-1 * m2 * r2 * (dot_th2 ** 2) * np.sin(th1 - th2) - (m1 + m2) * g * np.sin(th1),
         r1 * (dot_th1 ** 2) * np.sin(th1 - th2) - g * np.sin(th2)])

    ddot_th1, ddot_th2 = np.linalg.solve(M, K)
    return [dot_th1, ddot_th1, dot_th2, ddot_th2]

def double_pendulum_SAE_O1(t, state):
    th1, dot_th1, th2, dot_th2 = state

    M = [[(m1 + m2) * r1, m2 * r2],
         [r1,             r2]]
     
    K = [-(m1 + m2) * g * th1,
         -g * th2]

    ddot_th1, ddot_th2 = np.linalg.solve(M, K)
    return [dot_th1, ddot_th1, dot_th2, ddot_th2]

#O(th**3)
def double_pendulum_SAE_O3(t, state):
    th1, dot_th1, th2, dot_th2 = state
    
    M = [[(m1 + m2) * r1, m2 * r2 * (1 - (th1-th2)**2 / 2)],
         [r1 * (1 - (th1-th2)**2 / 2), r2]]

    K = [-m2 * r2 * dot_th2**2 * (th1-th2) - (m1 + m2) * g * (th1 - th1**3 / 6),
          r1 * dot_th1**2 * (th1-th2)      - g * (th2 - th2**3 / 6)]

    ddot_th1, ddot_th2 = np.linalg.solve(M, K)
    return [dot_th1, ddot_th1, dot_th2, ddot_th2]

#integrate
initial_state = [th1_inital, dot_th1_initial, th2_initial, dot_th2_initial]
t_eval = np.linspace(0, runtime, runtime*framerate)

#solve ivp
solution_real = solve_ivp(double_pendulum, (0, runtime), initial_state, t_eval=t_eval)
th1 = solution_real.y[0]
th2 = solution_real.y[2]

solution_SAE_O1 = solve_ivp(double_pendulum_SAE_O1, (0, runtime), initial_state, t_eval=t_eval)
th1_SAE_O1 = solution_SAE_O1.y[0]
th2_SAE_O1 = solution_SAE_O1.y[2]

solution_SAE_O3 = solve_ivp(double_pendulum_SAE_O3, (0, runtime), initial_state, t_eval=t_eval)
th1_SAE_O3 = solution_SAE_O3.y[0]
th2_SAE_O3 = solution_SAE_O3.y[2]

#ai generated code to trim arrays and account for errors in 
#solve_ivp as values approach unprocessable values
#code shortens array to shortest array so shapes match for plotting
n = min(solution_real.y.shape[1], solution_SAE_O1.y.shape[1], solution_SAE_O3.y.shape[1])
th1, th2 = solution_real.y[0][:n], solution_real.y[2][:n]
th1_SAE_O1, th2_SAE_O1 = solution_SAE_O1.y[0][:n], solution_SAE_O1.y[2][:n]
th1_SAE_O3, th2_SAE_O3 = solution_SAE_O3.y[0][:n], solution_SAE_O3.y[2][:n]

#convert to cartesian
x1 = r1 * np.sin(th1)
y1 = -1 * r1 * np.cos(th1)
x2 = r1 * np.sin(th1) + r2 * np.sin(th2)
y2 = -1 * r1 * np.cos(th1) - r2 * np.cos(th2)
        
x1_SAE_O1 = r1 * np.sin(th1_SAE_O1)
y1_SAE_O1 = -1 * r1 * np.cos(th1_SAE_O1)
x2_SAE_O1 = r1 * np.sin(th1_SAE_O1) + r2 * np.sin(th2_SAE_O1)
y2_SAE_O1 = -1 * r1 * np.cos(th1_SAE_O1) - r2 * np.cos(th2_SAE_O1)

x1_SAE_O3 = r1 * np.sin(th1_SAE_O3)
y1_SAE_O3 = -1 * r1 * np.cos(th1_SAE_O3)
x2_SAE_O3 = r1 * np.sin(th1_SAE_O3) + r2 * np.sin(th2_SAE_O3)
y2_SAE_O3 = -1 * r1 * np.cos(th1_SAE_O3) - r2 * np.cos(th2_SAE_O3)

#remap y values to origin at (0, r1+r2)
y1+= r1 + r2
y2 += r1 + r2
y1_SAE_O1 += r1 + r2
y2_SAE_O1 += r1 + r2
y1_SAE_O3 += r1 + r2
y2_SAE_O3 += r1 + r2

#find values for dp windows
x_min = np.min([x2, x2_SAE_O1, x2_SAE_O3])
x_max = np.max([x2, x2_SAE_O1, x2_SAE_O3])
y_min = np.min([y2, y2_SAE_O1, y2_SAE_O3])
y_max = np.max([y2, y2_SAE_O1, y2_SAE_O3])

#define plots
plot, ax = plt.subplots()

dp, = ax.plot(0, 0, marker='o', markerfacecolor='r', color='k', label='Real')
dp_SAE_O1, = ax.plot(0, 0, marker='o', markerfacecolor='g', color='k', label='SAE_O1')
dp_SAE_O3, = ax.plot(0, 0, marker='o', markerfacecolor='b', color='k', label='SAE_O3')

path, = ax.plot(0,0, color='r', alpha=0.2)
path_SAE_O1, = ax.plot(0,0, color='g', alpha=0.2)
path_SAE_O3, = ax.plot(0,0, color='b', alpha=0.2)

#set window
ax.set_aspect('equal')
ax.set_xlim(x_min - 0.1, x_max + 0.1)
if y_max < r1 + r2: ax.set_ylim(y_min - 0.1 , (r1 + r2) + 0.1)
else: ax.set_ylim(y_min - 0.1, y_max + 0.1)

ax.set_title('Real vs First Order SAE vs Third Order SAE')
ax.set_xlabel('X position in meters {m}')
ax.set_ylabel('Y position in meters {m}')
plot.legend(loc='lower right')

#animate
def animate(i):
    dp.set_xdata([0, x1[i], x2[i]])
    dp.set_ydata([r1+r2, y1[i], y2[i]])
    path.set_xdata(x2[:i])
    path.set_ydata(y2[:i])

    dp_SAE_O1.set_xdata([0, x1_SAE_O1[i], x2_SAE_O1[i]])
    dp_SAE_O1.set_ydata([r1+r2, y1_SAE_O1[i], y2_SAE_O1[i]])
    path_SAE_O1.set_xdata(x2_SAE_O1[:i])
    path_SAE_O1.set_ydata(y2_SAE_O1[:i])
    
    dp_SAE_O3.set_xdata([0, x1_SAE_O3[i], x2_SAE_O3[i]])
    dp_SAE_O3.set_ydata([r1+r2, y1_SAE_O3[i], y2_SAE_O3[i]])
    path_SAE_O3.set_xdata(x2_SAE_O3[:i])
    path_SAE_O3.set_ydata(y2_SAE_O3[:i])
    
    return dp, path, dp_SAE_O1, path_SAE_O1, dp_SAE_O3, path_SAE_O3

anim = FuncAnimation(plot, animate, frames=n * framerate, interval=1000/framerate, blit=True)
plt.show()