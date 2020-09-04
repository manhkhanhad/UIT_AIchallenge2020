import time, random
import numpy as np
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import draw_outputs, convert_boxes

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from PIL import Image
import os
import math
import json
import math
from shapely.geometry import Polygon

flags.DEFINE_string('classes', 'data/labels/obj.names', 'path to classes file')
flags.DEFINE_string('weights', 'weights/yolov3_toi.tf',
                    'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('video', 'data/video/test.mp4',
                    'path to video file or number for webcam)')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_integer('num_classes', 5, 'number of classes in the model')



#file_out.write("<fame> <label> <id> <xMax> <yMax> <xMin> <yMin> < -1> < -1> < -1>")
def load_ROI():
    video_name = os.path.splitext(FLAGS.video)[-2]
    path = os.getcwd()
    path = str(os.path.split(os.path.split(path)[0])[0])
    ZONE_PATH = os.path.join(path,"Data/{}/{}.json".format(video_name,video_name))
    with open(ZONE_PATH) as f:
        data = json.load(f)
    for shape in data["shapes"]:
        if shape["label"] == "zone":
            region = np.array(shape["points"])
    return region

"""
#KIEM TRA OBJECT CO TRONG MOT KHONG
def DienTichTamGiac(x,y,z):
    a = math.sqrt((x[0] - y[0])**2 + (x[1] - y[1])**2)
    b = math.sqrt((y[0] - z[0])**2 + (y[1] - z[1])**2)
    c = math.sqrt((z[0] - x[0])**2 + (z[1] - x[1])**2)
    p = (a+b+c)/2
    return math.sqrt(p*(p-a)*(p-b)*(p-c))

def DienTichDaGiac(region):
    s = 0
    region.astype(int)
    for i in range(-1,region.shape[0]-1):
        s += (region[i][0] - region[i+1][0])*((region[i][1] + region[i+1][1]))
    return abs(s/2)

def is_in_region(x,y,region):
    Tong_tam_giac = 0
    point = np.array([x,y])
    for i in range(0,region.shape[0]-1):
        Tong_tam_giac += DienTichTamGiac(region[i],region[i+1],point)

    Tong_tam_giac += DienTichTamGiac(region[0],region[-1],point)

    epsilon = 10**-6
    if (Tong_tam_giac - DienTichDaGiac(region)) <= epsilon:
        return True
    else:
        return False
#-----------------------------------------------
"""

def is_in_region(top_left,bot_right,region):
    x_min = top_left[0]
    x_max = bot_right[0]
    y_min = top_left[1]
    y_max = bot_right[1]
    bbox = Polygon([(x_min,y_min),(x_min,y_max),(x_max,y_max),(x_max,y_min)])
    pts = []
    for point in region:
        pts.append((point[0],point[1]))
    roi = Polygon(pts)
    if bbox.intersects(roi):
        return True
    else:
        return False

def class_to_classNumber(label):
    if label == 'loai_1':
        return 1
    if label == 'loai_2':
        return 2
    if label == 'loai_3':
        return 3
    if label == "loai_4":
        return 4
    if label == "di_bo":
        return 0

def main(_argv):
    region = load_ROI()


    # Definition of the parameters
    max_cosine_distance = 0.3  #Default = 0.5
    nn_budget = None
    nms_max_overlap = 0.8      #Default = 0.5 

    #initialize deep sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    if FLAGS.tiny:
        yolo = YoloV3Tiny(classes=FLAGS.num_classes)
    else:
        yolo = YoloV3(classes=FLAGS.num_classes)

    yolo.load_weights(FLAGS.weights)
    logging.info('weights loaded')

    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    logging.info('classes loaded')


    #CODE WRITE RESULT
    video_name = os.path.splitext(FLAGS.video)[-2]
    result = "tracking_result/{}_track.txt".format(video_name)
    file_out = open(result,'w')
    path = os.getcwd()
    path = str(os.path.split(os.path.split(path)[0])[0])
    vid_path = os.path.join(path,"Data/{}/{}.mp4".format(video_name,video_name))

    vid = cv2.VideoCapture(vid_path)

    out = None

    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))
    frame_index = -1 
    
    fps = 0.0
    count = 0 
    while True:
        _, img = vid.read()

        if img is None:
            break

        img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        img_in = tf.expand_dims(img_in, 0)
        img_in = transform_images(img_in, FLAGS.size)

        t1 = time.time()
        boxes, scores, classes, nums = yolo.predict(img_in)
        classes = classes[0]
        names = []
        for i in range(len(classes)):
            names.append(class_names[int(classes[i])])
        names = np.array(names)
        converted_boxes = convert_boxes(img, boxes[0])
        features = encoder(img, converted_boxes)    
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(converted_boxes, scores[0], names, features)]
        
        #initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # run non-maxima suppresion
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]        

        # Call the tracker
        tracker.predict()
        tracker.update(detections)
        frame_index = frame_index + 1
        print('FRAME: ',frame_index)
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            bbox = track.to_tlbr()
            class_name = track.get_class()
            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 1)
            #cv2.rectangle(img, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
            #cv2.putText(img, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
            x_cen = int((int(bbox[2]) + int(bbox[0]))/2)
            y_cen = int((int(bbox[3]) + int(bbox[1]))/2)

            if is_in_region((int(bbox[0]), int(bbox[1])),(int(bbox[2]), int(bbox[3])),region) == False:  #NGOAI ROI THI XOA
                track.delete_track()

            cv2.putText(img,"FRAME: "+ str(frame_index),(0,45),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,255,0),2)
            
            #GHI FILE TRACKING_RESULT theo chuan CountMovement
            bb_width = int(bbox[2]) - int(bbox[0])
            bb_height = int(bbox[3]) - int(bbox[1])
            diagonal = math.sqrt(bb_height**2 + bb_width**2)
            file_out.write("{},{},{},{},{},{},{},{},{}\n".format(frame_index,track.track_id,x_cen,y_cen,diagonal,-1.0,class_to_classNumber(str(class_name)),bb_width,bb_height))

        ### UNCOMMENT BELOW IF YOU WANT CONSTANTLY CHANGING YOLO DETECTIONS TO BE SHOWN ON SCREEN
        for det in detections:
            bbox = det.to_tlbr() 
            cv2.rectangle(img,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(0,255,0), 1)
        
        # print fps on screen 
        fps  = ( fps + (1./(time.time()-t1)) ) / 2
        cv2.putText(img, "FPS: {:.2f}".format(fps), (0, 30),
                          cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
        cv2.imshow('output', img)
        if FLAGS.output:
            out.write(img)
            #frame_index = frame_index + 1
            #list_file.write(str(frame_index)+' '+'\n')
            #if len(converted_boxes) != 0:
            #    for i in range(0,len(converted_boxes)):
            #        list_file.write(str(converted_boxes[i][0]) + ' '+str(converted_boxes[i][1]) + ' '+str(converted_boxes[i][2]) + ' '+str(converted_boxes[i][3]) + ' ')
            #        list_file.write('\n')

        # press q to quit
        if cv2.waitKey(1) == ord('q'):
            break
    vid.release()
    if FLAGS.output:
        out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
