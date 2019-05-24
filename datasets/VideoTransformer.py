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

def split_video_ffmpeg(video_path, output_path):

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    cmd = './ffmpeg -i ' + video_path + ' ' + output_path + '/%06d.jpg -loglevel quiet'
    os.system(cmd)

    # frame id 1 base -> 0 base
    frame_n = len(os.listdir(output_path))
    for fid in range(frame_n):
        org_frame_path = '%s/%06d.jpg' % (output_path, fid+1)
        new_frame_path = '%s/%06d.jpg' % (output_path, fid)
        os.renames(org_frame_path, new_frame_path)


if __name__ == '__main__':

    videoGroups = os.listdir(dataset_root)

    for videoGroup in videoGroups:
        videoGroupPath = dataset_root + videoGroup + "/"
        videoPaths = glob.glob(videoGroupPath + "*.mp4")
        for videoPath in videoPaths:
            jpgPath = videoPath[:-4]  # mp4 format
            if os.path.exists(jpgPath):
                continue
            os.mkdir(jpgPath, 755)
           # extractFrames(videoPath, jpgPath, 1)
            split_video_ffmpeg(videoPath, jpgPath)
        print("Finish group " + videoGroup)

    print("Finish")