from datetime import datetime
import sys
import tempfile
import cvlib    # high level module, uses YOLO model with the find_common_objects method
import cv2      # image/video manipulation, allows us to pass frames to cvlib
import os
import json


# these will need to be fleshed out to not miss any formats
IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.tiff', '.gif']
VID_EXTENSIONS = ['.mov', '.mp4', '.avi', '.mpg', '.mpeg', '.m4v', '.mkv']

# used to make sure we are at least examining one valid file
VALID_FILE_ALERT = False
# if an error is dectected, even once. Used for alerts
ERROR_ALERT = False
#used for alerts. True if human found once
HUMAN_DETECTED_ALERT = False

OUTPUT = {
    'logs':[],
    'thumbnail': None,
    'labels':[],
}

# function takes a file name(full path), checks that file for human shaped objects
# saves the frames with people detected into directory named 'save_directory'
def humanChecker(save_directory, yolo='yolov3', continuous=False, nth_frame=10, confidence=.65, gpu=False):

    # for modifying our global variarble VALID_FILE
    global VALID_FILE_ALERT

    # tracking if we've found a human or not
    is_human_found = False
    analyze_error = False
    is_valid = False
    video_labels = set()

    # we'll need to increment every time a person is detected for file naming
    person_detection_counter = 0
    inf = sys.stdin
    with tempfile.NamedTemporaryFile('wb') as tfile:
        OUTPUT['logs'].append(f"{datetime.now()} reading video from stdin")
        tfile.write(inf.buffer.read())
        video_file_name = tfile.name
        OUTPUT['logs'].append(f"{datetime.now()} wrote video to {video_file_name}")
        vid = cv2.VideoCapture(tfile.name)
        OUTPUT['logs'].append(f"{datetime.now()} got VideoCapture OBJ")
        # get approximate frame count for video
        frame_count = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        OUTPUT['logs'].append(f"{datetime.now()} got frame count")
        #make sure it's a valid video
        if frame_count > 0:
            VALID_FILE_ALERT = True
            is_valid = True
            OUTPUT['logs'].append(f'{datetime.now()} {frame_count} frames')
        else:
            is_valid = False
            analyze_error = True


        if is_valid:
            # look at every nth_frame of our video file, run frame through detect_common_objects
            # Increase 'nth_frame' to examine fewer frames and increase speed. Might reduce accuracy though.
            # Note: we can't use frame_count by itself because it's an approximation and could lead to errors
            for frame_number in range(1, frame_count - 6, nth_frame):

                OUTPUT['logs'].append(f'{datetime.now()} frame {frame_number}')
                # if not dealing with an image
                if os.path.splitext(video_file_name)[1] not in IMG_EXTENSIONS:
                    vid.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                    _, frame = vid.read()

                # feed our frame (or image) in to detect_common_objects
                try:
                    bbox, labels, conf = cvlib.detect_common_objects(frame, model=yolo, confidence=confidence, enable_gpu=gpu)                 
                except:
                    analyze_error = True
                    break
                video_labels.update(labels)

    return is_human_found, analyze_error, video_labels


#############################################################################################################################
if __name__ == "__main__":
    time_stamp = datetime.now().strftime('%m%d%Y-%H:%M:%S')

    # check for people
    human_detected, error_detected, labels =  humanChecker( time_stamp, yolo='yolov3',)

    OUTPUT['labels'] = [l for l in labels]            
    OUTPUT['logs'].append(f'{datetime.now()} {labels}')

    print(json.dumps(OUTPUT))
