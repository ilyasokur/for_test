

import cv2 as cv

cap = cv.VideoCapture("video.mp4")
length = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
print(length)

def cutVideo():
    i = 0
    video = cv.VideoCapture('video1.mp4')
    while True:
        ret, frame = video.read()
        c = cv.waitKey(50)
        if c == 27:
            break
        new_frame = frame[115:350, 210:445]
        if video.get(cv.CAP_PROP_POS_FRAMES) % 15 == 0:
            new_frame = cv.resize(new_frame, dsize=(119,119))
            cv.imwrite('imgs/' + str(i) + '.png', new_frame)
            print(str(i) + '.png')
            i = i + 1
cutVideo()
cv.destroyAllWindows()


