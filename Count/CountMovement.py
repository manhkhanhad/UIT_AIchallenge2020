import os,json
import numpy as np
from PIL import Image, ImageDraw
from collections import namedtuple
from enum import IntEnum, auto
from scipy.ndimage import gaussian_filter1d
from scipy.special import expit  # pylint: disable=no-name-in-module
from scipy.stats import linregress, norm
from absl import app, flags, logging
from absl.flags import FLAGS
import sys
import predict
root_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.insert(1,root_path)
from Visualize.visualize import visualize
import math
from predict import predictROI
from shapely.geometry import Polygon

FLAGS = flags.FLAGS
flags.DEFINE_string('video','007.mp4','path to video')
flags.DEFINE_boolean('visualize',False,'visualize result')
#parser = argparse.ArgumentParser()
#parser.add_argument("-v","--video",required=True,type=str,help="path to video")
#args = parser.parse_args()
#parser.add_argument('tracking','track_file/{}.txt'.format(video_name),'path to track result file')
#parser.add_argument('ROI','data/{}.txt'.format(video_name),'path to ROI file')
#parser.add_argument('MOI','data/{}.json'.format(video_name),'path to MOI file')

TrackItem = namedtuple('TrackItem', ['frame_id', 'obj_type', 'data'])

Event = namedtuple('Event', [
    'video_id', 'frame_id', 'movement_id', 'obj_type', 'confidence', 'track_id',
    'track'])

class ObjectType(IntEnum):
    '''
        Loại 1: xe 2 bánh như xe đạp, xe máy
        Loại 2: xe 4-7 chỗ như xe hơi, taxi, xe bán tải…
        Loại 3: xe trên 7 chỗ như xe buýt, xe khách
        Loại 4: xe tải, container, xe cứu hỏa
    '''
    loai_1 = 1
    loai_2 = 2
    loai_3 = 3
    loai_4 = 4
    nguoi = 0

    
#Config
IMG_HEIGHT = 720 
IMG_WIDTH = 1280 
FPS =  10
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



def getTrackItemByTrackId(track_id,path,region):
    track_items = []
    track_line = []
    index_line = -1
    with open(path,'r') as f:
        data = f.read().splitlines()
    for d in data:
        index_line+=1
        v = d.split(",")
        if int(v[1])==track_id:
            class_id = 1 if float(v[6]) == -1.0 else int(float(v[6]))
            confi = 1 if float(v[5]) == -1.0 else float(v[5])
            track_item = TrackItem(int(v[0]), class_id, (float(v[2]), float(v[3]), float(v[4]), confi))
            #BO SUNG PREDICT VI TRI ROI ROI:
            #kiem tra track_item[-1] co nam trong ROI khong neu khong thi chay predict
            #append vi tri va frame vua predict duoc vao
            track_items.append(track_item)
            track_line.append([int(v[0]),float(v[2]),float(v[3]),float(v[7]),float(v[8])])
            #print(track_id,track_item)


    #return track_items
    # bien phap chua chay tam thoi: do obj_id khong ton tai nen track_item tra ve laf rong, khi chay
    # lenh get_event se bi loi, nen tai day neu track_item == [] tra ve None
    # bien phap sua loi cai tien: Fix code file tracking sao cho stt obj luon tang dan, khong co khoang trong

    if track_items == []:
        return None
    else:
        
        last_track = track_items[-1]
        diagon = last_track[2][2]
        confident = last_track[2][3]
        last_frame = last_track[0]
        last_coordinates = track_line[-1]
        x_cen = last_coordinates[1]
        y_cen = last_coordinates[2]
        width = last_coordinates[3]
        height = last_coordinates[4]
        x_topLeft = x_cen - width/2
        y_topLeft = y_cen - height/2
        x_botRight = x_cen + width/2
        y_botRight = y_cen + height/2
        
        if is_in_region((x_topLeft,y_topLeft),(x_botRight,y_botRight),region) == True and len(track_line) >1:
            #print("CHAY PREDICT")
            predict_loc = predictROI(last_frame,np.array(track_line),region.astype(int))
            predict_frame, predict_x, predict_y = predict_loc.predictTime()
            if(predict_frame) == None:
                return track_items
            track_item = TrackItem(predict_frame, class_id, (predict_x, predict_y, diagon, confident))

            with open(path, "a") as file_object:
                file_object.write("{},{},{},{},{},{},{}\n".format(predict_frame,track_id,predict_x,predict_y,diagon,-1.0,class_id))
        

            track_items.append(track_item)
            #print(track_id,track_item)
            
        return track_items



def get_region_mask(region, height, width):
    img = Image.new('L', (width, height), 0)
    region = region.flatten().tolist()
    ImageDraw.Draw(img).polygon(region, outline=0, fill=255)
    mask = np.array(img).astype(np.bool)
    return mask


def get_track(track_items, region ,min_length = 0.3,stride=1, gaussian_std = 0.3,speed_window=1,min_speed=10):
    img_height= IMG_HEIGHT
    img_width= IMG_WIDTH
    fps= FPS
    min_length = max(3, min_length * fps)
    speed_window = int(speed_window * fps // 2) * 2
    init_frame_id = track_items[0].frame_id
    length = track_items[-1].frame_id - init_frame_id + 1
    if init_frame_id % 100 == 0:
        print("frame_id",init_frame_id)
    if length < min_length:
        return None
    if len(track_items) == length:
        interpolated_track = np.stack([t.data for t in track_items])
    else:
        interpolated_track = np.empty((length, len(track_items[0].data)))
        interpolated_track[:, 0] = -1
        for t in track_items:
            interpolated_track[t.frame_id - init_frame_id] = t.data
        for frame_i, state in enumerate(interpolated_track):
            if state[0] >= 0:
                continue
            for left in range(frame_i - 1, -1, -1):
                if interpolated_track[left, 0] >= 0:
                    left_state = interpolated_track[left]
                    break
            for right in range(frame_i + 1, interpolated_track.shape[0],1):
                if interpolated_track[right, 0] >= 0:
                    right_state = interpolated_track[right]
                    break
            movement = right_state - left_state
            #if(right-left != 0):
            ratio = (frame_i - left) / abs(right - left*0.99) #TRANH RIGHT - LEFT = 0
            #print("ratio",ratio)
            interpolated_track[frame_i] = left_state + ratio * movement
    if gaussian_std is not None:
        gaussian_std = gaussian_std * fps
        track = gaussian_filter1d(
            interpolated_track, gaussian_std, axis=0, mode='nearest')
    else:
        track = interpolated_track
    track = np.hstack([track, np.arange(
        init_frame_id, init_frame_id + length)[:, None]])
    speed_window = min(speed_window, track.shape[0] - 1)
    speed_window_half = speed_window // 2
    speed_window = speed_window_half * 2

    speed = np.linalg.norm(
        track[speed_window:, :2] - track[:-speed_window, :2], axis=1)
    speed_mask = np.zeros((track.shape[0]), dtype=np.bool)
    speed_mask[speed_window_half:-speed_window_half] = \
        speed >= min_speed
    speed_mask[:speed_window_half] = speed_mask[speed_window_half]
    speed_mask[-speed_window_half:] = speed_mask[-speed_window_half - 1]
    track = track[speed_mask]
    track_int = track[:, :2].round().astype(int)


    region_mask = get_region_mask(region, img_height, img_width)
    iou_mask = region_mask[track_int[:, 1], track_int[:, 0]]
    track = track[iou_mask]
    if track.shape[0] < min_length:
        return None
    return track

    
#Get movement_score
start_proportion_factor=1.5
def get_movement_heatmaps(movements, height, width):
    distance_heatmaps = np.empty((len(movements), height, width))
    proportion_heatmaps = np.empty((len(movements), height, width))
    xs, ys = np.meshgrid(np.arange(width), np.arange(height))
    points = np.stack([xs.flatten(), ys.flatten()], axis=1)
    for label, movement_vertices in movements.items():
        vectors = movement_vertices[1:] - movement_vertices[:-1]
        lengths = np.linalg.norm(vectors, axis=-1) + 1e-4
        rel_lengths = lengths / lengths.sum()
        vertex_proportions = np.cumsum(rel_lengths)
        vertex_proportions = np.concatenate([[0], vertex_proportions[:-1]])
        offsets = ((points[:, None] - movement_vertices[None, :-1])
                   * vectors[None]).sum(axis=2)
        fractions = np.clip(offsets / (lengths ** 2), 0, 1)
        targets = movement_vertices[:-1] + fractions[:, :, None] * vectors
        distances = np.linalg.norm(points[:, None] - targets, axis=2)
        nearest_segment_ids = distances.argmin(axis=1)
        nearest_segment_fractions = fractions[
            np.arange(fractions.shape[0]), nearest_segment_ids]
        distance_heatmap = distances.min(axis=1)
        proportion_heatmap = vertex_proportions[nearest_segment_ids] + \
            rel_lengths[nearest_segment_ids] * nearest_segment_fractions
        distance_heatmaps[label - 1, ys, xs] = distance_heatmap.reshape(
            height, width)
        proportion_heatmaps[label - 1, ys, xs] = proportion_heatmap.reshape(
            height, width)
    return distance_heatmaps, proportion_heatmaps

#default proportion_thres_to_delta=0.5, distance_base_size=4, distance_scale=5, start_period=0.3, start_thres=0.5,proportion_scale=0.8,distance_slope_scale=2,merge_detection_score=False,final=True):
def get_movement_scores(track, obj_type, movements, proportion_thres_to_delta=0.5, distance_base_size=4, distance_scale=5, start_period=0.3, start_thres=0.5,proportion_scale=0.8,distance_slope_scale=2,merge_detection_score=False,final=True):
    img_height= IMG_HEIGHT
    img_width= IMG_WIDTH
    fps=FPS

    #with open(os.path.join(zone_path)) as f:
    #    data = json.load(f)
    #movements = {int(shape['label']): np.array(shape['points']) for shape in data['shapes']}  #vvdfvfdvfdvfdvdfvvf
    #assert len(movements) == max(movements.keys())
 
    positions = track[:, :2].round().astype(int)
    diagonals = track[:, 2]
    detection_scores = track[:, 3]
    frame_ids = track[:, -1]

    distance_heatmaps, proportion_heatmaps =  get_movement_heatmaps(movements, img_height, img_width)

    distances = distance_heatmaps[:, positions[:, 1], positions[:, 0]]
    proportions = proportion_heatmaps[:, positions[:, 1], positions[:, 0]]

    distances = distances / diagonals[None]
    mean_distances = distances.mean(axis=1)
    x = np.linspace(0, 1, proportions.shape[1])
    distance_slopes = np.empty((len(movements)))
    proportion_slopes = np.empty((len(movements)))

    for movement_i in range(len(movements)):
        distance_slopes[movement_i] = linregress(
            x, distances[movement_i])[0]
        proportion_slopes[movement_i] = linregress(
            x, proportions[movement_i])[0]


    proportion_delta = proportions.max(axis=1) - proportions.min(axis=1)
    proportion_slopes = np.where(
        proportion_slopes >= proportion_thres_to_delta,
        proportion_delta, proportion_slopes)
    if obj_type == ObjectType.loai_1:
        distance_base_scale = min(
            1, distance_base_size / mean_distances.shape[0])
        distance_base = np.sort(mean_distances)[
            :distance_base_size].sum() * distance_base_scale
        score_1 = 1 - (mean_distances / distance_base) ** 2
    else:
        score_1 = expit(4 - mean_distances * distance_scale)

    proportion_factor = 1 / proportion_scale
    score_2 = proportion_factor * np.minimum(proportion_slopes, 1 / (proportion_slopes + 1e-8))
    start_period=start_period*fps
    if frame_ids[0] <= start_period and score_2.max() <= start_thres:
        score_2 *= start_proportion_factor
    score_3 = norm.pdf(distance_slopes * distance_slope_scale) / 0.4
    scores = np.stack([score_1, score_2, score_3], axis=1)
    if final:
        scores = np.clip(scores, 0, 1).prod(axis=1)
        if merge_detection_score:
            scores = scores * detection_scores.mean()
    return scores



def get_obj_type(track_items, track):
    active_frame_ids = set(track[:, -1].tolist())
    obj_types = [t.obj_type for t in track_items                       #B_U_G: obj_types = []
                 if t.frame_id in active_frame_ids]
    if (len(obj_types) == 0):
        return ObjectType.nguoi
    type_counts = np.bincount(obj_types)
    class_id = np.argmax(type_counts)

    #Bike:0, Car:1, Truck:2
    if class_id == 1:
        obj_type = ObjectType.loai_1
    elif class_id == 2:
        obj_type = ObjectType.loai_2
    elif class_id == 3:
        obj_type = ObjectType.loai_3
    elif class_id == 4:
        obj_type = ObjectType.loai_4
    elif class_id == 0:
        obj_type = ObjectType.nguoi

    return obj_type

def get_event(video_id, track_id, tracking_result_path, region, movement, stride=1,min_score=0.125,return_all_events=False):
    '''
        input:
        - (int)video_id -- the id of video tracking
        - (int)track_id -- the label of tracking to get movement result
        - (string)tracking_result_path -- the tracking result file path
        output:
          A movements result
          Event = namedtuple('Event', ['video_id', 'frame_id', 'movement_id', 'obj_type', 'confidence', 'track_id', 'track'])
    '''
    #Get track_id items
    track_items = getTrackItemByTrackId(track_id,tracking_result_path,region)

    if track_items is None:
        return None
    track = get_track(track_items,region)

    if track is None:
        return None

    obj_type = get_obj_type(track_items, track)
    frame_id = (track_items[-1][0] + 1) * stride

    #Get movement_scores
    movement_scores = get_movement_scores(track, obj_type, movement)

    max_index = movement_scores.argmax()
    max_score = movement_scores[max_index]
    if len(movement) == 1:
        min_score = 0
    if max_score < min_score:
        if not return_all_events:
            return None
        movement_id = 0
    else:
        movement_id = max_index + 1

    event = Event(video_id, frame_id, movement_id, obj_type, max_score, track_id, track_items)
    return event


def get_multi_event(video_id,st_id,en_id,tracking_result_path, region,movement):
    events = []
    for track_id in range(st_id,en_id+1):
        event = get_event(video_id,track_id,tracking_result_path, region,movement)
        if event is None:
            continue
        events.append(event)
    events.sort(key=lambda x: x.frame_id)
    return events
    
def create_submission(results,result_file,video_name):
    with open(result_file,'w') as f:
        for d in results:
            #print(d)
            f.write("{} {} {} {}\n".format(video_name,d.frame_id,d.movement_id,d.obj_type.value))
    print("the result is saved at " + result_file )


def main(_argv):
    video_name = os.path.splitext(FLAGS.video)[-2]
    TRACKING_RESULT_PATH = os.path.join(root_path,"Tracking/yolov3_deepsort/tracking_result/{}_track.txt".format(video_name))
    #ROI_PATH = os.path.join(root_path,"Data/{}/{}.txt".format(video_name,video_name))
    #MOI_PATH = os.path.join(root_path,"Data/{}/{}.json".format(video_name,video_name))
    ZONE_PATH = os.path.join(root_path,"data/test_data/{}.json".format(video_name))
    with open(TRACKING_RESULT_PATH, "rb") as file:
        file.seek(-2, os.SEEK_END)
        while file.read(1) != b'\n':
            file.seek(-2, os.SEEK_CUR) 
        trackid_en = int((file.readline().decode()).split(',')[1]) # trackid_en la track_id lon nhat trong file result


    video_id = 0
    trackid_st = 1

    #LOAD MOI and ROI
    region = []
    movement = {}
    with open(ZONE_PATH) as f:
        data = json.load(f)
    for shape in data["shapes"]:
        if shape["label"] == "zone":
            region = np.array(shape["points"])
        else:
            movement[ int(shape["label"][-2:]) ] = np.array(shape["points"]) 

    results = get_multi_event(video_id,trackid_st,trackid_en,TRACKING_RESULT_PATH,region,movement)
    results_path = os.path.join(root_path,'Result/{}_result.txt'.format(video_name))
    create_submission(results,results_path,video_name)


    if FLAGS.visualize == True:
        list_obj = []
        list_direction = []
        for event in results:
            list_obj.append(event.track_id)
            list_direction.append(event.movement_id)
        video_path = os.path.join(root_path,"Tracking/yolov3_deepsort/{}_result.avi".format(video_name))
        output_path = os.path.join(root_path,"Result/{}_visualize.avi".format(video_name))
        visualize(list_obj,list_direction,results_path,TRACKING_RESULT_PATH,video_path,output_path,region,movement)

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass