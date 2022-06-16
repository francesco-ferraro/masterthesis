#!/usr/local/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import os, sys



if False:
    data = np.loadtxt("./out_field.txt")
    part = np.loadtxt("./out_part.txt")
    
    print(data.shape)
    x = range(len(data[0]))
    
    for t in range(len(data)):
        if t%100!=0: continue
        plt.clf()
        plt.plot(data[t])
        plt.plot(x[1:-1], 100*part[t], ".")
        plt.ylim((-200, 200))
        plt.vlines((50-10, 50+10), -20, 20)
        plt.title(f"t = {t}")
        plt.draw()
        plt.pause(0.001)
        
    plt.show()

if True:
    dt = 0.01
    x0 = 5
    k = 0.1
    n_traj = 25000
    
    # data = np.load("./out.npy")
    # for datafile in ["out_R0.txt", "out_R4.txt"]:
    for datafile in ["script_test_free_1763813.out.txt"]:
        data = np.loadtxt(datafile)
        x = data[0]
        xsq = data[1]

        ts = np.arange(len(x))*dt


        var = xsq - x**2
        stddev = np.sqrt(var/n_traj)

        plt.plot(ts, x)
        plt.fill_between(ts, x-stddev, x+stddev, alpha=0.2)
 
    theory = x0*np.exp(-k*ts)       
    plt.plot(ts, theory)
    # plt.plot(ts[1:], ts[1:]**(-3/2))
    
    # plt.legend(("R = 0", "R = 4"))
    # plt.xscale("log")
    plt.yscale("log")
    plt.show()

if False:
    k = 0.1
    x0 = 5
    T = 100
    n_traj = 1000000
    
    dts = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
    
    for dt in dts:
        print(f"(* dt = {dt} *)")
        t_steps = int(T/dt)
        outfile = f"linear_vs_exp/lin_dt{dt}_ntraj{n_traj}_tsteps{t_steps}.npy"
        cmd = f"./test_single_particle_lin {dt} {n_traj} {t_steps} {outfile}"
        os.system(cmd + " &> /dev/null")
    
        data = np.load(outfile)
        x = data[0]
        xsq = data[1]
    
        ts = np.arange(len(x))*dt
        theory = x0*np.exp(-k*ts)

        var = xsq - x**2
        stddev = np.sqrt(var/n_traj)

        # plt.plot(ts, x)
        # plt.fill_between(ts, x-stddev, x+stddev, alpha=0.2)
        # plt.plot(ts, theory)
        # plt.yscale("log")
        # plt.show()
    
        err = np.sqrt(np.sum((x[1:]-theory[1:])**2/var[1:])/len(x[1:]))
    
        print(f"{{{dt:.4f},{err:.10f}}},")

if False:
    data = np.load("out.npy")
    x = data[0]
    xsq = data[1]
    
    for X in x:
        if X != 0:
            dx = np.abs(X)
            break

    bins = np.arange(-6-0.5*dx, 6+1.5*dx, dx)
    n, bins, patches = plt.hist(x, density=True, bins=bins)

    bins = (bins[1:] + bins[:-1])/2
    th = np.exp(-((bins/3)**2-1)**2)
    th /= np.sum(th)*(bins[1]-bins[0])
    plt.plot(bins, th, ".-")

    print(f"dx = {dx}, t_steps = {len(x)}, err = {np.sqrt(np.sum((n-th)**2))/len(bins)}")

    plt.show()