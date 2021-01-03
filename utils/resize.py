import cv2
import os

if __name__ == "__main__":
    dir = "/data/private/NeRF_Data/mine/manequinn_small/images"
    for f in os.listdir(dir):
        full_dir = os.path.join(dir, f)
        img = cv2.imread(full_dir)
        print(img.shape)
        img = cv2.resize(img, (3160, 1640), interpolation=cv2.INTER_CUBIC)
        print(cv2.imwrite(full_dir, img))
        print(img.shape)