# from ..config import *

import nengo
import numpy as np

import warnings

# This is for controlling the PTU 
# positive means coordinates to the left, and 
# above the centre of the frame

class NengoController:
    def __init__(self, 
            k_pan=np.array([1.,0.,0.]), 
            k_tilt=np.array([1.,0.,0.]),
            target_pan=64,
            target_tilt=64,
            depth=0.0,
            simulator_dt=0.01, # 0.01
            ):
        self.k_pan = k_pan
        self.k_tilt = k_tilt
        # need to initialize the actual pan and tilt coords
        # to something meaningful that will not cause 
        # dramatic swings when the controller is first turned
        # on.
        self.tilt = target_tilt# m
        self.pan = target_pan# m
# 
        self.target_pan = target_pan # m
        self.target_tilt = target_tilt # m
        self.simulator_dt = simulator_dt
        self.construct_network()
        pass

    def construct_network(self):
        self.model = nengo.Network()

        with self.model:
            tilt_inp = nengo.Node(lambda t: self.tilt)
            pan_inp = nengo.Node(lambda t: self.pan)

            pan_target = nengo.Node(lambda t: self.target_pan)
            tilt_target = nengo.Node(lambda t: self.target_tilt)

            ## We are doing a PID controller where the target is to have
            ## 0 error on the tiltontal component of the saliancy map 
            ## maximum, and maintain a fixed distance for depth.

            ## proportional error on tilt
            tilt_err_prop = nengo.Ensemble(n_neurons=50, dimensions=1, radius = 300) #neurons were 50
            nengo.Connection(
                    tilt_target, 
                    tilt_err_prop,
                    transform=-self.k_tilt[0]
                )
            nengo.Connection(
                    tilt_inp,
                    tilt_err_prop,
                    transform=self.k_tilt[0]
                )
            ## proportional error on pan


            ## The population should encode
            ## k_pan_prop * (pan - target_pan)
            pan_err_prop = nengo.Ensemble(n_neurons=50, dimensions=1, radius = 300)
            nengo.Connection(
                    pan_target,
                    pan_err_prop,
                    transform=-self.k_pan[0]
                )
            nengo.Connection(
                    pan_inp, 
                    pan_err_prop,
                    transform=self.k_pan[0]
                )

            self.p_tilt = nengo.Probe(tilt_err_prop)
            self.p_pan = nengo.Probe(pan_err_prop)
        ## end with Model

        self.sim = nengo.Simulator(self.model, dt=self.simulator_dt)



    def __call__(self, salmax_coords, sim_run_time=1.):
        # normalise the value between -1 and 1
        self.pan = salmax_coords[1]
        self.tilt = salmax_coords[0]

        self.sim.run(sim_run_time) # in seconds

        ## This computes the average of the last 10 msec of 
        ## simulated nengo data
        ## Unless otherwise specified the nengo simulator uses a 1msec
        ## time step and the probe data is indexed in milliseconds
        ## TODO: determine optimal time window size, if not 10 msec
        data_tilt = -np.mean(self.sim.data[self.p_tilt][-10:])
        data_pan = -np.mean(self.sim.data[self.p_pan][-10:])

        return (data_tilt,data_pan)
