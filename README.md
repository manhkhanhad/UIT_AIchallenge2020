# Multi-Class Multi-Movement Vehicle Counting
# Quick Start
0. Install libraries 
```
pip install -r requirements.txt
```
1. Load weight  
Download YOLOv3 weight which train with 4 types of vehicle in VietNam (type 1: bike, motorbike; type 2: car; type 3: bus, type 4: truck, container) and put in to data\test_data. [link dowload](https://drive.google.com/file/d/1nzoJrKI2Q26GfqiCK0yJ39YQaW6Gepyp/view?usp=sharing)  
Convert YOLOv3 weight to .tf file with cmd:
```
python Tracking/yolov3_deepsort/load_weight.py
```
2. Put data into data/test_data
3. Run  
To run whole your data, --visualize=True if you want to save visualize 
```
python run_all.py --visualize=True
```
If you just run on a video
```
python run.py --video=<video_name.mp4> --visualize=True
```
Your result saved in Result folder

# Example
![Imgur](https://imgur.com/l9Oj8XG)
# The related repos
[DeepSORT](https://github.com/theAIGuysCode/yolov3_deepsort)  
[YOLOv3](https://arxiv.org/abs/1804.02767)  
[Zero-VIRUS*](https://github.com/Lijun-Yu/zero_virus)  
