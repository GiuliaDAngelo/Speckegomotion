# Egomotion Suppression for Event-Based Cameras

**Contributors**:
- Giulia D'Angelo
- Alexander Hadjiivanov



![pipeline](https://github.com/GiuliaDAngelo/Speckegomotion/blob/main/images/egomotion.png)


## Overview

Egomotion refers to the movement of a camera relative to its environment.
In the context of event-based cameras, which capture changes in pixel intensity rather than full frames, egomotion introduces significant noise.
This noise can obscure the detection of relevant motion signals in the scene, such as moving objects, making it difficult to differentiate between background motion caused by the camera's movement and dynamic objects of interest.

### The Challenge of Egomotion Suppression

Event-based cameras operate asynchronously, capturing the dynamic range of changes at a very high temporal resolution. However, when a camera moves, it generates a large number of events due to changes in the scene's background. This overwhelming number of background events, resulting from egomotion, makes it hard to isolate the actual motion of objects.

### Key Objectives

The goal of egomotion suppression is to:
1. **Remove or reduce background noise** caused by camera movement.
2. **Enhance the detection of real, meaningful motion**, such as moving objects within the scene.

This suppression allows for better performance in tasks like object tracking, scene understanding, or navigation, particularly in robotics and autonomous systems, where distinguishing between self-motion and external motion is crucial.

### Approach in this Project

This project uses a neural network-based method to process event-based data and suppress egomotion by:
- **Analyzing motion patterns** across multiple frames to distinguish consistent background motion (egomotion) from sporadic object motion.
- **Applying running statistics** (mean, variance, standard deviation) to adaptively learn and subtract egomotion from the event data.
- **Using a multi-scale pyramid structure**, where different resolutions are processed to ensure egomotion suppression is effective at various scales of motion.

By applying these techniques, the project aims to effectively suppress egomotion while preserving meaningful motion signals from the scene.

## Data Download

You can download the necessary event-based data for this project from the following link:

- [Download Event-Based Data](https://www.dropbox.com/scl/fo/rbxtjaar1evhe6vnvrp7e/AC-g2YvSKrYdmbHqVuExflM?rlkey=jrdj1qmwnj9gkdyqmdwz9aguo&st=ucpl34q4&dl=0)

## Installation

1. **Clone the repository**:
    ```sh
    git clone https://github.com/GiuliaDAngelo/Speckegomotion.git
    cd Speckegomotion
    ```

2. **Create a virtual environment** (optional but recommended):
    ```sh
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

4. **Download the event-based data** (if not already done):
    ```markdown
    You can download the necessary event-based data for this project from the following link:
    [Download Event-Based Data](https://www.dropbox.com/scl/fo/rbxtjaar1evhe6vnvrp7e/AC-g2YvSKrYdmbHqVuExflM?rlkey=jrdj1qmwnj9gkdyqmdwz9aguo&st=ucpl34q4&dl=0)
    ```

5. **Run the main script**:
    ```sh
    python main.py
    ```

