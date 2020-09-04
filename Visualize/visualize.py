import cv2
import numpy as np
import json
import csv
import matplotlib.pyplot as plt
from absl import app, flags, logging
import os
import sys


root_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.insert(1,root_path)


def visualize(list_obj,list_direction,results_file,track_result,video_in,output,region,movements):
    
    #ROI_PATH = os.path.join(root_path,"Data/{}/{}.txt".format(video_name,video_name))
    #MOI_PATH = os.path.join(root_path,"Data/{}/{}.json".format(video_name,video_name))

    video_name = os.path.splitext(video_in)[-2]
    # Xu li so loai xe
    
    result = open(results_file)
    reader = csv.reader(result, delimiter=' ')
    kq = []
    frame = 0
    loai_1 = 0
    loai_2 = 0
    loai_3 = 0
    loai_4 = 0
    for row in reader:
        #kq.append([row[1],row[2]])
        while(len(kq) == 0 or frame < int(row[1])):
            kq.append([frame,loai_1,loai_2,loai_3,loai_4])
            frame += 1
        if(int(row[3]) == 1):
            loai_1 += 1
        if(int(row[3]) == 2):
            loai_2 += 1
        if(int(row[3]) == 3):
            loai_3 += 1
        if(int(row[3]) == 4):
            loai_4 += 1
        kq.append([frame,loai_1,loai_2,loai_3,loai_4])
        frame += 1
    for frame in kq:
        print(frame)

    # Ve BBOX
    
    track_file = open(track_result)
    annot = []
    reader = csv.reader(track_file, delimiter=',')
    sortedlist = sorted(reader, key=lambda row: int(row[0]), reverse=False)
    for row in sortedlist:
        annot.append([int(row[0]), int(row[1]), float(row[2]), float(row[3]), float(row[4]), int(row[6])])
    for a in annot:
        print(a)
    annot_index = 0
    vi_tri_track = [None]*len(list_obj)



    # Xu li Video
    vid = cv2.VideoCapture(video_in)
    # Dinh dang video ra
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    codec = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output, codec, fps, (width, height))

    frame = -1
    while True:
        _,img = vid.read()
        if img is None:
            break
        
        cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        frame += 1
        cv2.polylines(img,[region.astype(np.int32)],True,(255,0,0),1) # ROI


        #MOI
        for direct in movements:
            cv2.polylines(img,[movements[direct].astype(np.int32)],False,(0,0,255))
            x_text = int(movements[direct][-1][0])
            y_text = int(movements[direct][-1][1])
            cv2.putText(img,str(direct),(x_text,y_text),0,0.75,(187,0,255),2)
            

        cv2.rectangle(img,(width-200,10),(width,90),(140, 140, 140),-1)
                 
        cv2.putText(img,"loai 1: " + str(kq[frame][1]),(width-190,25),cv2.QT_FONT_NORMAL,0.75,(255,255,255))
        cv2.putText(img,"loai 2: " + str(kq[frame][2]),(width-190,45),cv2.QT_FONT_NORMAL,0.75,(255,255,255))
        cv2.putText(img,"loai 3: " + str(kq[frame][3]),(width-190,65),cv2.QT_FONT_NORMAL,0.75,(255,255,255))
        cv2.putText(img,"loai 4: " + str(kq[frame][4]),(width-190,85),cv2.QT_FONT_NORMAL,0.75,(255,255,255))

        #BBOX
        #Init color for bbox
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        while (annot_index < len(annot) and annot[annot_index][0] == frame ) :
            if(annot[annot_index][1] in list_obj):
                x = int(annot[annot_index][2])
                y = int(annot[annot_index][3])
                track_id = list_obj.index(annot[annot_index][1]) + 1   # De in ra theo thu tu obj tang dan
                object_type = int(annot[annot_index][5])        
                direction = list_direction[track_id-1]
                color = colors[int(track_id) % len(colors)]
                color = [i * 255 for i in color]
                cv2.circle(img,(x,y),10,color,-1)
                cv2.putText(img,str(track_id),(x,y-10),0,0.75,color,2)
                cv2.putText(img,str(object_type),(x+10,y+10),0,0.5,(0,0,255),2)
                cv2.putText(img,str(direction),(x-10,y+10),0,0.5,(0,255,255),2)
                #Trackline
                if vi_tri_track[track_id-1] == None:
                    vi_tri_track[track_id-1] = [[x,y]]
                else:
                    vi_tri_track[track_id-1].append([x,y])
                    pts = np.array(vi_tri_track[track_id-1],np.int32)
                    pts.reshape((-1, 1, 2))
                    cv2.polylines(img,[pts],False,(255,255,0),2)
            annot_index += 1
        print("frame: ", frame)
        #if (int(annot[annot_index][0]) > frame):
        #    continue
        
        cv2.imshow('output', img)
        if cv2.waitKey(1) == ord('q'):
                break

        out.write(img)

    vid.release()
    out.release()
    cv2.destroyAllWindows()
    

'''
if __name__ == '__main__':
    track_result = "result_convert.txt"
    video_in = '007.mp4'
    output = "visualize_007.avi"
    #visualize(list_obj,results_file,track_result,video_in,output)
    try:
        app.run(visualize(list_obj,results_file,track_result,video_in,output))
    except SystemExit:
        pass
'''