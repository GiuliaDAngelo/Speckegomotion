# Wandering Around: A Bioinspired Approach to Foveation Through Object Motion Sensitivity

**Authors:**  
Giulia D’Angelo<sup>1,*</sup>, Victoria Clerico<sup>2</sup>, Chiara Bartolozzi<sup>3</sup>, Matej Hoffmann<sup>1</sup>, P. Michael Furlong<sup>4</sup>, Alexander Hadjiivanov<sup>5</sup>  

---

**Affiliations:**  
1. Department of Cybernetics, Faculty of Electrical Engineering, Czech Technical University, Prague, Czech Republic  
2. IBM Research Europe, Zurich, Switzerland  
3. Event-Driven Perception for Robotics, Italian Institute of Technology, Genoa, Italy  
4. National Research Council of Canada & Systems Design Engineering, University of Waterloo, Canada  
5. The Netherlands eScience Center, Netherlands  

---

**Corresponding Author:**  
Giulia D’Angelo – [giulia.dangelo@fel.cvut.cz](mailto:giulia.dangelo@fel.cvut.cz)



![pipeline](https://github.com/GiuliaDAngelo/Speckegomotion/blob/main/WanderingAround.png)

---

## YouTube Video Demonstration:

[![Watch the video](https://img.youtube.com/vi/enXUAffZGC8/0.jpg)](https://www.youtube.com/watch?v=enXUAffZGC8&ab_channel=GiuliaD%27Angelo)


## Abstract

Active vision provides a dynamic and robust approach to visual perception, presenting a compelling alternative to the static and passive nature of feedforward architectures commonly employed in state-of-the-art computer vision for robotics. Traditional approaches often rely on training large datasets and demand significant computational resources.
Selective vision, inspired by biological mechanisms, enables active vision to direct agents' focus to salient areas within their visual field through selective attention mechanisms, processing only Regions of Interest (ROIs). This targeted approach significantly reduces computational demands while preserving real-time responsiveness.
Event-based cameras, inspired by the mammalian retina, further enhance this capability by capturing asynchronous changes in a scene, facilitating efficient, low-latency visual processing. To distinguish objects in motion while the event-based camera itself is also in motion within a dynamic scene, the agent requires an object motion segmentation model to accurately detect and foveate on the target.

Integrating event-based sensors with neuromorphic algorithms represents a paradigm shift, leveraging Spiking Neural Networks (SNNs) to parallelize computation and adapt to dynamic environments.
This work introduces a fully spiking Convolutional Neural Network (sCNN) bioinspired architecture designed for selective attention through object motion segmentation. The proposed system actively foveates on the current ROI, generating events through a Dynamic Vision Sensor (DVS) integrated into the Speck neuromorphic hardware mounted on a Pan-Tilt unit.
The system, characterized on ideal gratings and benchmarked against the Event Camera Motion Segmentation Dataset (EVIMO) and the Event-Assisted Low-Light Video Object Segmentation Dataset (LLE-VOS), achieves 96\% SSIM and 82.2\% IoU in multi-object motion segmentation. It also achieves 88.8\% object detection accuracy in office scenarios and 89.8\% in challenging indoor and outdoor low-light conditions. Its learning-free design ensures robustness and adaptability to diverse perceptual scenes, making it a reliable solution for real-time robotics applications in autonomous systems.
