import cv2
import numpy as np

threshold = 100

video = cv2.VideoCapture('DashCam/DashCam1.mp4')
chk, img= video.read()
sampler = 5
blob = cv2.SimpleBlobDetector_create()
fno = 0

def filterColors(image):
    # Define thresholds
    red_l = np.array([0, 80, 80])
    red_u = np.array([20, 255, 255])
    yellow_l = np.array([5, 55, 120])
    yellow_u = np.array([35, 255, 255])
    green_l = np.array([60, 100, 50])
    green_u = np.array([120, 200, 200])

    # Pull out pixels in threshold
    mask_red = cv2.inRange(image, red_l, red_u)
    mask_yellow = cv2.inRange(image, yellow_l, yellow_u)
    mask_green = cv2.inRange(image, green_l, green_u)

    # Find the contours based on the masked images
    #contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #contours_yellow, _ = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #contours_green, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return mask_red, mask_yellow, mask_green

def drawContours(image,contour):
    for contour in contour:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)


while chk:
    fno += 1
    if fno % sampler == 0:
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        m_red, m_ylw, m_grn = filterColors(hsv_img)
        #cv2.imshow('red mask',m_red)
        #cv2.imshow('yellow mask',m_ylw)
        cv2.imshow('green mask',m_grn)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel = np.ones((9, 9), np.uint8)
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        ret, thresh = cv2.threshold(tophat, threshold, 185, cv2.THRESH_BINARY)
        dist_transform = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
        ret, markers = cv2.connectedComponents(np.uint8(dist_transform))
        watershed = cv2.watershed(img, markers)
        keypoints = blob.detect(tophat)
        tophat_blob = cv2.drawKeypoints(tophat, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        #drawContours(tophat_blob,c_red)
        #drawContours(tophat_blob,c_ylw)
        #drawContours(tophat_blob,c_grn)
        cv2.imshow('default',img)
        cv2.waitKey(1)
    chk, img = video.read()