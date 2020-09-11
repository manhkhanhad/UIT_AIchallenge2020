# Multi-Class Multi-Movement Vehicle Counting
Ho Chi Minh AI challenge 2020
# Quick Start
0. Install libraries 
```
pip install -r requirements.txt
```
1. Load weight  
Download [YOLOv3 weight](https://drive.google.com/file/d/1nzoJrKI2Q26GfqiCK0yJ39YQaW6Gepyp/view?usp=sharing) which train with 4 types of vehicle in VietNam (type 1: bike, motorbike; type 2: car; type 3: bus, type 4: truck, container) and put into data\test_data.   
Convert YOLOv3 weight to .tf file with cmd:
```
python Tracking/yolov3_deepsort/load_weight.py
```
2. Put data into data/test_data
3. Run  
To run all video, --visualize=True if you want to save visualize 
```
python run_all.py --visualize=True
```
To run on a video
```
python run.py --video=<video_name.mp4> --visualize=True
```
Your result saved in Result folder

# Example
[![Imgur](Visualize/visualize.gif)](https://youtu.be/RuBIjW7oBpM)
# The related repos
[DeepSORT](https://github.com/theAIGuysCode/yolov3_deepsort)  
[YOLOv3](https://arxiv.org/abs/1804.02767)  
[Zero-VIRUS*](https://github.com/Lijun-Yu/zero_virus)  
