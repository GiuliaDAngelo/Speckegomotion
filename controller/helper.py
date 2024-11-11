from controller.nengo_controller import NengoController


_nengo_controller_obj = None

def run_controller(salmax_coords, target_coords):
    global _nengo_controller_obj
    if _nengo_controller_obj is None:
        ## Create a singleton object for the nengo controller.
        ## parameters are PID gains for horizontal axis of saliency map
        ## PID gains for depth controller
        ## target horizontal position of max and target depth for object
        _nengo_controller_obj = NengoController(
                target_pan=target_coords[0],
                target_tilt=target_coords[1],
        )
    ### end if
    return _nengo_controller_obj(salmax_coords, target_coords)
