# Hand Gesture Recognizer (Real-time with Mediapipe)

**Author:** Dharshaan M  

## 📖 Project Overview
This project is a real-time hand gesture recognizer built using **Python**, **OpenCV**, and **Mediapipe**.  
It detects **four gestures**:
- ✋ Open Palm  
- ✊ Fist  
- ✌ Peace (V-sign)  
- 👍 Thumbs Up  

Supports recognition of **two hands simultaneously**.

---

## Technology Justification

- **Mediapipe (Google):** Provides robust, lightweight, real-time hand landmark detection with high accuracy.  
- **OpenCV:** Used for webcam access, image pre-processing, and drawing bounding boxes/labels.  
- **NumPy:** For geometric calculations (angles, distances) to determine finger states.  

---

## Gesture Logic Explanation

The detection is based on **21 hand landmarks** provided by Mediapipe.  
We calculate **angles and relative distances** between landmarks to classify gestures.

1. **Open Palm:** Four or more fingers extended (angles > 165°).  
2. **Fist:** No fingers extended.  
3. **Peace (V-sign):** Index + Middle fingers extended and spread apart, others folded.  
4. **Thumbs Up:** Only the thumb extended upwards relative to the wrist.  


## 📜 License
This project is for educational purposes only.  
