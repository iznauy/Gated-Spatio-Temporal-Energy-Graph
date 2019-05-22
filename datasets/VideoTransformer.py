import cv2
import glob
import os

dataset_root = "./test/"

def extractFrames(videoPath, picPath, freq):
    cap = cv2.VideoCapture(videoPath)
    numFrame = -1
    while True:
        numFrame += 1
        res, image = cap.read()
        if not res:
            break
        if numFrame % freq == 0:
            cv2.imwrite(picPath + "{:06d}".format(numFrame) + ".jpg", image)


if __name__ == '__main__':

    videoGroups = os.listdir(dataset_root)

    for videoGroup in videoGroups:
        videoGroupPath = dataset_root + videoGroup + "/"
        videoPaths = glob.glob(videoGroupPath + "*.txt")
        for videoPath in videoPaths:
            jpgPath = videoPath[:-4] + "/" # mp4 format
            if os.path.exists(jpgPath):
                continue
            os.mkdir(jpgPath, 755)
            extractFrames(videoPath, jpgPath, 1)
        print("Finish group " + videoGroup)

    print("Finish")