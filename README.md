# Drones for Computer Vision Applications — Team 7

## 📌 Overview
This project was developed during the **Drone Summer School at the University of Southern Denmark** by **Team 7**:  
**Muhammet Bahadır Mutlu, Einar Benjamin Jensen, Anne Kirstine Rosenkvist Kaad, and David Blumbach**.  

Our goal was to use drones and computer vision techniques to capture geo-tagged images and automatically detect/count animal targets.

---

## 🎯 Project Goals
- Plan and execute autonomous flights at ~20 m altitude using GPS waypoints.
- Capture images and GPS coordinates at each waypoint.
- Use **color detection** (HSV bounds) to filter grass background.
- Apply **edge detection** and **Hu moments** to identify and count animal shapes.
- Save detection results and visual overlays.

---

## 🛠️ Main Features
1. **Autonomous data collection** with waypoint missions.
2. **Image preprocessing** — convert to HSV, threshold, and remove background.
3. **Feature extraction** with Hu moments.
4. **Counting & reporting** — save CSV summaries and detection overlays.

