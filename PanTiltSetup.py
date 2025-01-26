import serial
import time
import random


'''
PN * Minimum Pan position is -3090
PX * Maximum Pan position is 3090
TN * Minimum Tilt position is -907
TX * Maximum Tilt position is 604
'''


# Define the serial port and settings
serial_port = '/dev/tty.usbserial-FTGZO55F'
baud_rate = 9600

# Define pan and tilt range values
# pan_range = (-3090, 3090)  # Replace with actual pan range of your device if different
# tilt_range = (-907, 604)  # Replace with actual tilt range of your device if different

pan_range = (-400, 400)  # Replace with actual pan range of your device if different
tilt_range = (-400, 400)  # Replace with actual tilt range of your device if different

# Function to send a command to the serial device
def send_command(ser, command):
    ser.write(command.encode('utf-8'))
    time.sleep(1)  # Small delay to allow the device to process the command

random_mov = 20


try:
    with serial.Serial(serial_port, baud_rate, timeout=1) as ser:
        # Give some time to establish connection
        time.sleep(2)

        # # Send the initial "R" command to reset the pan and tilt unit, wait until the end of calibration
        # send_command(ser, 'R\n')
        # time.sleep(30)

        # Send random pan and tilt commands
        for _ in range(random_mov):  # Number of random movements
            # Generate random pan and tilt angles
            pan_angle = random.randint(*pan_range)
            tilt_angle = random.randint(*tilt_range)

            # Format the command (assuming the device uses "PP<angle>" and "TP<angle>")
            # Send the pan and tilt commands
            send_command(ser, f'PU\n')
            send_command(ser, f'PP{pan_angle}\n')

            send_command(ser, f'TU\n')
            send_command(ser, f'TP{tilt_angle}\n')

            send_command(ser, f'A\n')

            # Print commands sent for verification
            print(f"Sent pan command: {pan_angle}")
            print(f"Sent tilt command: {tilt_angle}")

            # Optionally, read and print any response from the device after each command
            response = ser.readline().decode('utf-8').strip()
            if response:
                print(f"Response from device: {response}")

            # Delay between movements
            time.sleep(0.5)

except serial.SerialException as e:
    print(f"Error: Could not open serial port {serial_port}. {e}")
