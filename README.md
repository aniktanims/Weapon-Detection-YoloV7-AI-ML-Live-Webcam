Weapon Detection Using YOLOv7
This project is a real-time weapon detection system utilizing the YOLOv7 object detection algorithm. It processes live webcam feeds to identify and locate weapons within the video stream, aiming to enhance security measures through prompt threat detection.

Features
Real-Time Detection: Processes live video streams from a webcam to detect weapons instantly.
High Accuracy: Employs the YOLOv7 algorithm, known for its precision in object detection tasks.
User-Friendly Interface: Provides a clear visual display, highlighting detected weapons in the video feed.
Installation
Clone the Repository:

bash
Copy
Edit
git clone https://github.com/aniktanims/Weapon-Detection-YoloV7-AI-ML-Live-Webcam.git
Navigate to the Project Directory:

bash
Copy
Edit
cd Weapon-Detection-YoloV7-AI-ML-Live-Webcam
Install Dependencies: Ensure you have Python installed. Install the required packages using pip:

bash
Copy
Edit
pip install -r requirements.txt
Download YOLOv7 Weights: Download the pre-trained YOLOv7 weights and place them in the project directory. You can obtain the weights from the official YOLOv7 repository.

Usage
Run the Detection Script:

bash
Copy
Edit
python detect.py
This script will access your webcam and start the real-time weapon detection. Detected weapons will be highlighted in the video feed.

Acknowledgments
This project utilizes the YOLOv7 algorithm for object detection. Special thanks to the developers and contributors of YOLOv7 for their work.

License
This project is licensed under the MIT License.
