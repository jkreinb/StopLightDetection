import cv2
import numpy as np

threshold = 100

video = cv2.VideoCapture('DashCam/Footage 3.mov')
chk, img= video.read()
sampler = 15

params = cv2.SimpleBlobDetector_Params()

params.filterByArea = 1
params.minArea = 15
params.maxArea = 250
params.minDistBetweenBlobs = 15


params.filterByColor = 0
params.blobColor = 255

params.filterByCircularity = 1
params.minCircularity = .75

params.filterByConvexity = 0
params.minConvexity = 0.40

params.filterByInertia = 0
params.minInertiaRatio = .05

blob = cv2.SimpleBlobDetector_create(params)
fno = 0

def filterColors(image):
    # Define thresholds for red lower & upper ([0, 100, 40])
    #([10, 150, 100])
    #([170, 100, 40])
    #([180, 150, 100])
    red_l1 = np.array([0, 100, 40])
    red_l2 = np.array([10, 255, 255])
    red_u1 = np.array([170, 100, 40])
    red_u2 = np.array([180, 255, 255])
    
    # Define thresholds for yellow lower & upper
    yellow_l = np.array([20, 60, 60])
    yellow_u = np.array([55, 255, 255])
    
    # Define thresholds for green lower & upper
    green_l = np.array([60, 160, 40])
    green_u = np.array([140, 255, 255])

    # Pull out pixels within threshold
    mask_red1 = cv2.inRange(image, red_l1, red_l2)
    mask_red2 = cv2.inRange(image,red_u1,red_u2)
    mask_yellow = cv2.inRange(image, yellow_l, yellow_u)
    mask_green = cv2.inRange(image, green_l, green_u)
    mask_red = mask_red1 +mask_red2
    return mask_red, mask_yellow, mask_green

def filterPixels(mask): # Erodes then dilates to help reduce single pixel noise
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.dilate(mask, kernel,iterations=2)
    mask = cv2.erode(mask, kernel)
    return mask

def detectDrawKeypoints(original, mask, color): # draws keypoints in corresponding colors
    keypoints = blob.detect(mask)
    
    for iter in range(len(keypoints)):
        x_p = np.int(keypoints[iter].pt[0])
        y_p = np.int(keypoints[iter].pt[1])
        size = 2* np.int(keypoints[iter].size)
        mask = cv2.rectangle(mask, (x_p-size,y_p-size),(x_p+size,y_p+size),color,3)
        original = cv2.rectangle(original, (x_p-size,y_p-size),(x_p+size,y_p+size),color,3)
    return original, mask
    


while chk:
    fno += 1 # Counts frame
    if fno % sampler == 0: # Only sample frames based on sampler
        original_h , original_w, chnls = img.shape
        
        cropped_img = img[0:np.int(original_h/2)+25,:] # Crops image to reduce reflections + noise from below traffic light height
        
        
        cropped_img = cv2.normalize(cropped_img, None, 0, 50, cv2.NORM_MINMAX, dtype=-1) # Normalize image to help reduce lighting effects
        cropped_blurred_img = cv2.GaussianBlur(cropped_img, (3,3),cv2.BORDER_DEFAULT) # Blur cropped images (3,3)
        cropped_hsv_img = cv2.cvtColor(cropped_blurred_img, cv2.COLOR_BGR2HSV) # Convert cropped image to HSV
        cv2.imshow('cropped image', cropped_img)
        m_red, m_ylw, m_grn = filterColors(cropped_hsv_img)
        
        img = cv2.normalize(img, None, 0, 150, cv2.NORM_MINMAX, dtype=-1) # Normalize image to help reduce lighting effects
        blur_img = cv2.GaussianBlur(img, (5,5),cv2.BORDER_DEFAULT) # Blur images
        hsv_img = cv2.cvtColor(blur_img, cv2.COLOR_BGR2HSV) # Convert to HSV
        # m_red, m_ylw, m_grn = filterColors(hsv_img) # Call filterColors to get masks of red/yellow/green colors

        m_grn = filterPixels(m_grn) # Call filterPixels per channel to help reduce noise
        m_ylw = filterPixels(m_ylw)
        m_red = filterPixels(m_red)

        # Draw keypoints for each match in their respective colors
        img, m_grn = detectDrawKeypoints(img,m_grn,(0,255,0))
        img, m_yellow = detectDrawKeypoints(img,m_ylw,(0,255,255))
        img, m_red = detectDrawKeypoints(img,m_red,(0,0,255))

        # Scaling for image
        width = int(img.shape[1] * .5)
        height = int(img.shape[0] * .5)
        dim = (width,height)
        img=cv2.resize(img,dim, interpolation = cv2.INTER_AREA)
        m_grn = cv2.resize(m_grn, dim, interpolation= cv2.INTER_AREA)
        #cv2.imshow('green mask',m_grn)
        cv2.imshow('red mask',m_red)
        #cv2.imshow('yellow mask',m_ylw)
        
        cv2.imshow('default',img)
        cv2.waitKey(0)
    chk, img = video.read()