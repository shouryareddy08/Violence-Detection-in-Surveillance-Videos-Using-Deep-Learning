# 🎥 Real-Time Object & Video Detection for Smart Security Surveillance

A real-time intelligent surveillance system that detects and tracks objects (like backpacks, bottles, and suitcases), identifies abandoned objects, and sends instant alerts via email with snapshots and sound notifications — enhancing safety in public and private spaces.

Submitted by
2410030322 V Manish Reddy
2410030396 K Manish Reddy
2410030367 S Shourya Reddy
2410030184 G Harish
---


## 🚀 Features

-  **Real-Time Object Detection** – Detects multiple objects simultaneously using **YOLOv3**.  
-  **Abandoned Object Detection** – Automatically triggers an alert if an object remains unattended for a defined duration *(default: 10 seconds)*.  
-  **Instant Email Alerts** – Sends a notification with the detected frame to the registered email address.  
-  **Sound Alerts** – Plays an audible alert *(via `alert.wav`)* when an unattended object is detected.  
-  **Web Interface** – Flask-based live video stream with options to set alert email or trigger manual alerts.  
-  **Custom Object Tracking** – Tracks position and movement; differentiates between attended and unattended objects.  
-  **Smart Detection Logic** – Avoids false positives if a person is near the object.  
-  **Image Logging** – Saves alert frames for future review in the `/static/` folder.  

---

## 💡 Use Cases

-  **Public Places (Airports, Malls, Stations)** – Detect unattended baggage or suspicious items in real time.  
-  **Educational Institutions** – Monitor hallways, labs, and campuses for unattended objects.  
-  **Hospitals / Offices** – Enhance facility safety and reduce security risks.  
-  **Smart Homes / IoT Systems** – Integrate as part of intelligent home surveillance.  

---

## 🛠 Tech Stack

| Component | Technology Used |
|------------|----------------|
| **Programming Language** | Python |
| **Framework** | Flask |
| **Computer Vision** | OpenCV + YOLOv3 |
| **Alert System** | SMTP (Email) + Sound |
| **Frontend** | HTML, CSS (via Flask Templates) |





