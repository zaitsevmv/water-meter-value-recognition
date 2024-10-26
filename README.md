# water-meter-value-recognition
C++ OpenCV lib that detects digits on water meters

# Detecting water meter

![image](https://github.com/user-attachments/assets/93b07c90-7965-40d4-b436-2237c89eebe8)
![image](https://github.com/user-attachments/assets/c9ed4245-af0b-44c2-bfa8-1ecbd8626c45)
![image](https://github.com/user-attachments/assets/4b2a9d58-ebbd-4681-86aa-005f9bdafb07)

Dataset: 1244 pictures

TP (90% of digits inside circle): 1208

Wrong circle: 23

Nothing: 13

# Detecting rectangle with digits

![image](https://github.com/user-attachments/assets/96d5952e-640b-4341-8ee7-da5f81ae87fb)

FROC accuracy 0.065 (IoU=0.1)

FROC accuracy 0.02 (IoU=0.2)

# Recognising digits (using DenseNet_BiLSTM_CTC)

Detection and recognition accuracy: 0.22 (0.247 for left side, 0.18 for right side)(F1 metric).

# Results

Overall accuarcy - 0.014. Should not be used by any means: ML is easier and more accurate.  
