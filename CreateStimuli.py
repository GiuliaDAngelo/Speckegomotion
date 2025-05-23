from psychopy import visual, core, event  # import some libraries from PsychoPy
import cv2
import numpy as np
import os

def createfld(path):
    # Check if the folder exists, and create it if it doesn't
    if not os.path.exists(path):
        os.makedirs(path)
        print('Folder created')
    else:
        print('Folder already exists')

# Paths
exp =  'objego'
# exp =  'ego'
# exp =  'onlyobj'

one_grat = False
#Parameters stimuli
sf = 0.3  # spatial frequency [0.2, 1, 4]
sf_small = 3
speed = 0.13 # speed in cycles per frame
small_speed = 0.05
duration = 16  # duration in seconds


name_exp = exp+'sf'+str(sf)+'sp'+str(speed)+'sf'+str(sf_small)+'sp'+str(small_speed)
name_exp = name_exp.replace(".", "")

respath = '/Users/giuliadangelo/workspace/code/IEBCS/data/video/speedstimuli/'+name_exp+'/'
createfld(respath)

# Create a window with larger dimensions to ensure stimuli fit
square = [350, 350]  # Window size

mywin = visual.Window(square, monitor="testMonitor", units="deg")

# Create some stimuli
grating = visual.GratingStim(win=mywin, mask='circle', size=9, pos=[0, 0], sf=sf)
grating_small = visual.GratingStim(win=mywin, mask='circle', size=2, pos=[0, 0], sf=sf_small)

# Define the crop region (center portion of the window)
crop_width_fraction = 0.7  # Fraction of the window's width to capture (50% in this case)
crop_height_fraction = 0.5  # Fraction of the window's height to capture (30% in this case)
# Calculate the crop dimensions based on the fractions
crop_width = int(mywin.size[0] * crop_width_fraction)
crop_height = int(mywin.size[1] * crop_height_fraction)
# Calculate the starting point for the crop (centered)
x_start = int((mywin.size[0] - crop_width) // 2)
y_start = int((mywin.size[1] - crop_height) // 2)

# Draw the stimuli and update the window
clock = core.Clock()
frames = []

if one_grat:
    while clock.getTime() < duration:
        if exp == 'onlyobj':
            grating = grating_small
            grating.setPhase(small_speed, '+')
        else:
            grating.setPhase(speed, '+')  # advance phase
        grating.draw()
        mywin.flip()
        frame = mywin.getMovieFrame(buffer='front')  # Capture frame from front buffer
        frames.append(frame)
        event.clearEvents()
else:
    while clock.getTime() < duration:
        grating.setPhase(speed, '+')  # advance phase
        grating_small.setPhase(small_speed, '+')
        grating.draw()
        grating_small.draw()
        mywin.flip()
        frame = mywin.getMovieFrame(buffer='front')  # Capture frame from front buffer
        frames.append(frame)
        event.clearEvents()

# Save frames as a video using OpenCV
os.chdir(respath)
output_file = name_exp+".mp4"
fps = 1000

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' for .mp4 format
video_writer = cv2.VideoWriter(output_file, fourcc, fps, (crop_width, crop_height))

for frame in frames:
    # Convert the PsychoPy frame (Pillow image) to a numpy array
    frame_np = np.array(frame)
    # Crop the central portion
    frame_cropped = frame_np[y_start:y_start+crop_height, x_start:x_start+crop_width]
    # Convert to BGR for OpenCV
    frame_bgr = cv2.cvtColor(frame_cropped, cv2.COLOR_RGB2BGR)
    video_writer.write(frame_bgr)

video_writer.release()

# Cleanup
mywin.close()
core.quit()

print(f"Video saved as {output_file}")
