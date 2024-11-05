import dv_processing as dv

cameras = dv.io.discoverDevices()
#
# print(f"Device discovery: found {len(cameras)} devices.")
# for camera_name in cameras:
#     print(f"Detected device [{camera_name}]")

try:
    cameras = dv.io.discoverDevices()
except Exception as e:
    print(f"Error discovering devices: {e}")
    cameras = []