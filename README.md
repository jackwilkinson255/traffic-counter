# YOLOv5 Traffic Counter with PyTorch Data Engineering Project

<p align="center">
  <img src="data/videos/detected.gif" alt="animated" />
</p>

<p align="center">
  <b>Object detection and counting</b>
</p>

This project collects data gathered from [YOLOv5](https://github.com/ultralytics/yolov5) to detect objects and the 
[SORT](https://github.com/abewley/sort) algorithm to track those objects over different frames so they can be counted



## Object Detection




## Tracking Algorithm





## Data Engineering

## ETL Pipeline

###Extraction:
Data is extracted from an Arduino microcontroller conncted to two passive infrared (PIR) sensors which detect motion. 
These are placed on either side of the cat flap we can then tell whether the cat is coming in or going out.
This data is sent from the Arduino to a Raspberry Pi via serial connection. 

###Transform:
The Raspberry Pi (running main.py) receives the status of the motion sensors. Knowing the order in which 
these motion sensors were triggered, we can detect if the is "inside" or "outside". We add a small delay to 
allow the cat to get through its flap before detecting a false reading. 

###Load: 
The data from the Raspberry Pi is then converted to json format and published through an MQTT connection
every 15 minutes. We then add a rule in IoT Core which scans for messages from a specified topic and then  
stores the device data into a DynamoDB table with a timestamp partition key.





