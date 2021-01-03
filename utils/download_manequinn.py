from pytube import YouTube
import cv2
import os


if __name__ == "__main__":
    data_path = "/data/private/MannequinChallenge/textfiles/train"
    for txt_file in os.listdir(data_path):
        with open(os.path.join(data_path, txt_file)) as f:
            url = f.readline().strip()

    frames = YouTube(url).filter(subtype='mp4').all()

    #* source: https://stackoverflow.com/questions/33311153/python-extracting-and-saving-video-frames
    vidcap = cv2.VideoCapture('big_buck_bunny_720p_5mb.mp4')
    success, image = vidcap.read()
    count = 0
    while success:
        cv2.imwrite("frame%d.jpg" % count, image)  # save frame as JPEG file
        success, image = vidcap.read()
        print('Read a new frame: ', success)
        count += 1