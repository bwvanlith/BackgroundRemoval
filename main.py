import cv2
import cv2.cv2
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os

#open webcam
cap = cv2.VideoCapture(0)

#set width and height of the video capture(being 3 and 4, set to 640x480)
cap.set(3, 640)
cap.set(4, 480)

#increase frame rate
cap.set(cv2.CAP_PROP_FPS, 60)

#selfie segmentation object
segmentor = SelfiSegmentation(1)

#writing the fps on screen
fpsReader = cvzone.FPS()

# loading single background image to a fixed variable (used in imgOut =)
imgBg = cv2.imread("Images/background_01.jpg")

# get list of all background images in Images folder
listImg = os.listdir("Images")
# make empty list called imgList
imgList =[]
# adding all images in folder to a list called imgList
for imgPath in listImg:
    img = cv2.imread(f'Images/{imgPath}')
    imgList.append(img)
indexImg = 0




# While webcam is on
while True:
    success, img = cap.read()

    #run the segmentor with color background threshold (1 = full removal, 0 = no removal)
    #imgOut = segmentor.removeBG(img, (255,0,0), threshold=0.8)

    # run the segmentor with images background threshold (1 = full removal, 0 = no removal), with ImgList being the background image
    imgOut = segmentor.removeBG(img, imgList[indexImg], threshold=0.85)

    #stack original and segmented images (names, colums, scale)
    imgStacked = cvzone.stackImages([img,imgOut],2,1)

    #add the fps to the imgStacked footage (return the fps and the stackedimage)
    fps, imgStacked = fpsReader.update(imgStacked, color=(0,0,255))

    #open images normal and segmented
    cv2.imshow("Image", imgStacked)

    # show two seperate images instead of a stacked one
    # cv2.imgshow("Image", img)
    # cv2.imgshow("Image Segmented", imgOut)

    key = cv2.waitKey(1)

    # switching the background image with keypress
    if key == ord('a') and indexImg > 0:
        indexImg -= 1
    elif key == ord('d') and indexImg < len(imgList)-1:
        indexImg += 1
    elif key == ord('q'):
        break

    print(indexImg)
