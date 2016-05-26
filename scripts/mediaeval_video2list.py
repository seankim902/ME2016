import os
import cv2

data_folder_path = "/media/joyful/HDD/media_eval/LIRIS-ACCEDE-data"

def video_sampling(file_path, freq):
    cap = cv2.VideoCapture(file_path)
    while not cap.isOpened():
        cap = cv2.VideoCapture(file_path)
        cv2.waitKey(100)
        print "Wait for the header"

    total_frame = cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
    print total_frame
    selected_frame_list = []
    for i in range(1, freq+1):
        selected_frame = (int)(total_frame/freq * i -1)
        selected_frame_list.append(selected_frame)    

    print selected_frame_list
    pos_frame = cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
    current_index = 0
    while True:
        flag, frame = cap.read()
        if flag:
            # The frame is ready and already captured
            # cv2.imshow('video', frame)
            pos_frame = cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
            
            if pos_frame == selected_frame_list[current_index]:
                current_index += 1
                jpg_name = "/%06d.jpg" % (current_index)
                cv2.imwrite(frame_folder_path+jpg_name, frame)
                print str(pos_frame)+" frames selected"
            if( current_index == len(selected_frame_list) ):
                current_index = len(selected_frame_list) - 1
                                
        else:
            # The next frame is not ready, so we try to read it again
            cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, pos_frame-1)
            print "frame is not ready"
            # It is better to wait for a while for the next frame to be ready
            cv2.waitKey(100)

        if cv2.waitKey(1) == 27:
            break
        if cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES) == cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT):
            # If the number of captured frames is equal to the total number of frames,
            # we stop
            break


def show_video(file_path):
    cap = cv2.VideoCapture(file_path)
    while not cap.isOpened():
        cap = cv2.VideoCapture(file_path)
        cv2.waitKey(100)
        print "Wait for the header"

    pos_frame = cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
    while True:
        flag, frame = cap.read()
        if flag:
            # The frame is ready and already captured
            cv2.imshow('video', frame)
            pos_frame = cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
            print str(pos_frame)+" frames"
        else:
            # The next frame is not ready, so we try to read it again
            cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, pos_frame-1)
            print "frame is not ready"
            # It is better to wait for a while for the next frame to be ready
            cv2.waitKey(100)

        if cv2.waitKey(10) == 27:
            break
        if cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES) == cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT):
            # If the number of captured frames is equal to the total number of frames,
            # we stop
            break
            
#show_video(file_path)
#os.mkdir(data_folder_path)
#os.mkdir(frame_folder_path)
#video_sampling(file_path, 16)

f = open('media_eval.txt','r')
with open('media_eval.txt','r') as f:
    lines = f.read().splitlines()

print len(lines)

for file_name in lines:
    frame_folder_path = data_folder_path + "/input/" + file_name
    print frame_folder_path
    os.mkdir(frame_folder_path)
    file_path = data_folder_path + "/data/" + file_name + ".mp4"
    video_sampling(file_path, 16)
                    