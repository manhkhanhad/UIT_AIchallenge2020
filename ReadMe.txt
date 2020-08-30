1. Dua du lieu vao folder Data, Load_weight(*)
2. cd den thu muc UIT-AI-Challenge2020
3. De chay toan bo qua trinh: python run.py --video=<ten_video.mp4> --visualize=True (de tao file video ket qua)
4. Ket qua o file result
Hoac chay tung module
- Chay Tracking: python Tracking/yolov3_deepsort/object_tracker.py --video= (ten video.mp4)
- Chay CountMovement: python Count/CountMovement.py --video=(ten video.mp4) --visualize=True (de tao file video ket qua)



* Load_weight:
1. Tai file weight 
2. Dua file weight vao "UIT_AIchallenge2020\Tracking\yolov3_deepsort\weights"
3. Sua ten file weight (line 7 file load_weight.py o folder yolov3_deepsort)
4. cmd: python load_weights.py
5. Sua ten file weight (line 25 file object_tracker o folder yolov3_deepsort)
