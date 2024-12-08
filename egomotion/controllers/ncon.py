# --------------------------------------
import nengo

# --------------------------------------
import numpy as np

# --------------------------------------
import matplotlib.pyplot as plt


class NengoController:
    """
    # This is for controlling the PTU
    # positive means coordinates to the left, and
    # above the centre of the frame
    """

    # REVIEW: This singleton is not needed.
    # Instead, there is an instance inside the hub.
    # controller = None

    # @staticmethod
    # def run_controller(salmax_coords, target_coords, k_pan, k_tilt):
    #     if NengoController.controller is None:
    #         ## Create a singleton object for the nengo controller.
    #         ## parameters are PID gains for horizontal axis of saliency map
    #         ## PID gains for depth controller
    #         ## target horizontal position of max and target depth for object
    #         NengoController.controller = NengoController(
    #             target_pan=target_coords[0],
    #             target_tilt=target_coords[1],
    #             k_pan=k_pan,
    #             k_tilt=k_tilt,
    #         )
    #     ### end if
    #     # REVIEW: This is odd - the __call__ signature is different
    #     return NengoController.controller(salmax_coords, target_coords)

    def __init__(
        self,
        k_pan: np.ndarray = np.array([1.0, 0.0, 0.0]),
        k_tilt: np.ndarray = np.array([1.0, 0.0, 0.0]),
        target_pan: int = 64,
        target_tilt: int = 64,
        depth: float = 0.0,
    ):
        self.k_pan = k_pan
        self.k_tilt = k_tilt
        # need to initialize the actual pan and tilt coords
        # to something meaningful that will not cause
        # dramatic swings when the controller is first turned on.
        # REVIEW: These the same as the targets - is that the intention?
        self.pan = target_pan  # m
        self.tilt = target_tilt  # m
        # REVIEW: What is this?
        self.depth = depth

        self.target_pan = target_pan  # m
        self.target_tilt = target_tilt  # m
        self.construct_network()

    def construct_network(self):
        self.model = nengo.Network()

        with self.model:
            tilt_inp = nengo.Node(lambda t: self.tilt)
            pan_inp = nengo.Node(lambda t: self.pan)

            pan_target = nengo.Node(lambda t: self.target_pan)
            tilt_target = nengo.Node(lambda t: self.target_tilt)

            # We are doing a PID controller where the target is to have
            # 0 error on the tiltontal component of the saliancy map
            # maximum, and maintain a fixed distance for depth.

            # proportional error on tilt
            tilt_err_prop = nengo.Ensemble(n_neurons=50, dimensions=1)
            nengo.Connection(tilt_target, tilt_err_prop, transform=-self.k_tilt[0])
            nengo.Connection(tilt_inp, tilt_err_prop, transform=self.k_tilt[0])
            # proportional error on pan

            # where is pan_err_pop defined? and tilt_err_pop?

            # The population should encode
            # k_pan_prop * (pan - target_pan)
            pan_err_prop = nengo.Ensemble(n_neurons=50, dimensions=1)
            nengo.Connection(pan_target, pan_err_prop, transform=-self.k_pan[0])
            nengo.Connection(pan_inp, pan_err_prop, transform=self.k_pan[0])

            self.p_tilt = nengo.Probe(tilt_err_prop)
            self.p_pan = nengo.Probe(pan_err_prop)
        # end with Model

        self.sim = nengo.Simulator(self.model)

    def test(
        self,
        actual: np.ndarray = None,
        target: np.ndarray = None,
    ):
        """
        Test the Nengo controller with some coordinates.
        Args:
            actual (np.ndarray):
                Actual coordinates (ground truth)

            target (np.ndarray):
                The target coordinates (requested)
        """
        # actual = np.array([20, 30])
        # target = np.array([64, 64])

        # REVIEW: Should these be the other way around (so run(target, actual))?

        result = NengoController.run(actual, target)
        return result

    def __call__(
        self,
        salmax_coords: np.ndarray,
        depth: float,
        sim_run_time: float = 5.0,
    ):
        """
        Run the controller with the saliency map coordinates.

        Args:
            salmax_coords (np.ndarray):
                The coordinates of the saliency map.

            depth (float):
                REVIEW: Is this needed? It's not used anywhere.

            sim_run_time (float, optional):
                Simulation time.
                Defaults to 5.0.

        Returns:
            tuple[float, float]:
                The pan and tilt angles.
        """

        # REVIEW: The depth and the horizontal coordinate
        # are not used anywhere.
        self.horiz_coordinate = salmax_coords[1]
        self.depth = depth

        self.sim.run(sim_run_time)  # in seconds

        # This computes the average of the last 10 msec of
        # simulated nengo data
        # Unless otherwise specified the nengo simulator uses a 1msec
        # time step and the probe data is indexed in milliseconds
        # TODO: determine optimal time window size, if not 10 msec
        data_tilt = np.mean(self.sim.data[self.p_tilt][-10:])
        data_pan = np.mean(self.sim.data[self.p_pan][-10:])

        return (data_pan, data_tilt)
