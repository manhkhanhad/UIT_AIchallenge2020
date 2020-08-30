from absl import app, flags, logging
from absl.flags import FLAGS
import os

FLAGS = flags.FLAGS
flags.DEFINE_string('video','007','video name')
flags.DEFINE_boolean('visualize',False,'visualize result')

#video_path = "Data/{}/{}.mp4".format(FLAGS.video,FLAGS.video)
def main(__argv):
    root_path = os.getcwd()
    video_name = os.path.splitext(FLAGS.video)[-2]
    #os.system("cd Tracking/yolov3_deepsort")
    os.chdir("Tracking/yolov3_deepsort")
    os.system("python object_tracker.py --video={}.mp4 --output={}_result.avi".format(video_name,video_name))


    os.chdir(os.path.join(root_path,"Count"))
    if FLAGS.visualize == True:
        os.system("python CountMovement.py --video={}.mp4 --visualize=True".format(video_name))
    else:
        os.system("python CountMovement.py --video={}.mp4 --visualize=False".format(video_name))

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
