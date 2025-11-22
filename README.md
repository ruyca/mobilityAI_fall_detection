# mobilityAI_fall_detection
Real-time vision AI system for detecting falling objects on metro tracks using Raspberry Pi, developed for the Siemens Mobility Challenge at the Austrian-Mexican hackathon.

## Model Selection 
YOLO11n (Nano)

**Key Advantages:**

- Latest technology: Released in 2024, represents state-of-the-art optimization
- 37% less complexity than YOLOv8 while maintaining similar accuracy
- Excellent Python support via the Ultralytics library
- Easy deployment: Simple pip install and 3-line Python code to run
- Well-documented: Extensive documentation and community support
- Pre-trained weights: Can detect 80 COCO classes out of the box
- Easy to customize: Simple fine-tuning for your specific falling object scenarios

**Performance Metrics:**

- Speed: 80-100ms inference time on Pi 5
- FPS: 10-12 frames per second
- Accuracy: 39.5% mAP50-95
- Parameters: 2.6M
- Size: ~5MB

**Real-world Performance:**

- Can detect objects in real-time for security/safety applications
- Low latency suitable for immediate alert generation
- Robust detection of various object types (luggage, backpacks, bottles, etc.) 

