import os
from absl import app, flags, logging
import glob

def main(__argv):
    for file in glob.glob("data/test_data/*.mp4"):
        name = os.path.splitext(file)[-2]
        name = name.split('\\')[-1]
        print(name)
        os.system("python run.py --video={}.mp4 --visualize=True".format(name))
    
    os.system("python processing.py")

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass