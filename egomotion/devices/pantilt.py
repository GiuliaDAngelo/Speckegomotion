# --------------------------------------
import numpy as np

# --------------------------------------
import serial

# --------------------------------------
import time

# --------------------------------------
import random

# --------------------------------------
from egomotion.conf import logger


class PTU:
    """
    Pan-Tilt Unit (PTU) controller.

    PN * Minimum Pan position is -3090
    PX * Maximum Pan position is 3090
    TN * Minimum Tilt position is -907
    TX * Maximum Tilt position is 604
    """

    def __init__(
        self,
        port: str = "/dev/tty.usbserial-FTGZO55F",
        rate: int = 9600,
        pan_range: np.ndarray = None,
        tilt_range: np.ndarray = None,
        timeout: int = 1,
        verbose: bool = True,
    ):

        # Serial port
        self.port = port

        # Baud rate
        self.rate = rate

        # Connection timeout
        self.timeout = timeout

        # Extra logging
        self.verbose = verbose

        # Replace with actual pan range of your device if different
        if pan_range is None:
            pan_range = [-3090, 3090]
        self.pan_range = np.array(pan_range, dtype=np.int32)

        if tilt_range is None:
            tilt_range = [-907, 604]
        self.tilt_range = np.array(tilt_range, dtype=np.int32)

        self.serial = serial.Serial(self.port, self.rate, self.timeout)

    # Function to send a command to the serial device
    def send_command(
        self,
        command: str,
        wait: float = None,
    ):
        """
        Send a command to the unit.

        Args:
            command (str):
                The command as a string.

            wait (str):
                The amount of time to wait for the device
                to process the command.
        """

        self.serial.write(command.encode("utf-8"))
        if wait:
            time.sleep(wait)

    def test(
        self,
        count: int = 20,
        warmup: float = 2.0,
        rest: int = 0.5,
    ):
        """
        Test the PTU by performing some random moves.

        Args:
            count (int, optional):
                Number of moves. Defaults to 20.

            warmup (float, optional):
                Warm-up time. Defaults to 2.0.

            rest (int, optional):
                Delay between moves. Defaults to 0.5.
        """

        try:

            # Give some time to establish connection
            time.sleep(warmup)

            # REVIEW: Is this still needed?
            # Send the initial "R" command to reset the pan and tilt unit, wait until the end of calibration
            # send_command(ser, 'R\n')
            # time.sleep(30)

            # Send random pan and tilt commands
            for _ in range(count):  # Number of random movements

                # Generate random pan and tilt angles
                pan_angle = random.randint(*self.pan_range)
                tilt_angle = random.randint(*self.tilt_range)

                # Format the command (assuming the device uses "PP<angle>" and "TP<angle>")
                # Send the pan and tilt commands

                response = self.move(pan_angle, tilt_angle)

                if response and self.verbose:
                    # Read and print any response from the device after each command
                    logger.debug(f"PTU '{self.port}' | Response: {response}")

                # Delay between movements
                time.sleep(rest)

        except serial.SerialException as e:
            logger.debug(
                f"PTU '{self.port}' | Error: Could not open serial port '{self.port}'."
            )
            raise e

    def move(
        self,
        pan_angle: int,
        tilt_angle: int,
    ) -> str:
        """
        Move the unit to the supplied pan / tilt angles.

        Args:
            pan_angle (int):
                Pan angle.

            tilt_angle (int):
                Tilt angle.

        Returns:
            str:
                The response from the unit.
        """

        if self.verbose:
            logger.debug(
                f"PTU '{self.port}' | Pan angle: '{pan_angle}' | Tilt angle: {tilt_angle}"
            )

        pan_command = f"PP{pan_angle}\n"
        tilt_command = f"TP{tilt_angle}\n"

        # Send the pan and tilt commands
        self.send_command(f"PU\n")
        self.send_command(pan_command)
        self.send_command(f"TU\n")
        self.send_command(tilt_command)
        self.send_command(f"A\n")

        return self.serial.readline().decode("utf-8").strip()

    def reset(self):
        """
        Reset the unit to angles (0,0)
        """

        self.move(0, 0)
