# python -m py_compile stitch.py
# import the necessary packages
from pyimagesearch.panorama import Stitcher
import imutils
import cv2

'''
    Stitch Function
Usage:  Stitches the images together using the Stitcher class and crops final image
Input:  imageA - some image
        imageB - some other image
        (images should interesect or function will not work)
Output: Returns a tuple of the stitched image and the visualization
'''
def stitch(imageA,imageB):
    imageA = cv2.imread(imageA)
    imageB = cv2.imread(imageB)
    imageA = imutils.resize(imageA, width=400)
    imageB = imutils.resize(imageB, width=400)

    stitcher = Stitcher()
    (result, vis) = stitcher.stitch([imageA, imageB], showMatches=True)

    #crop on blank space
    img = result
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    _,thresh = cv2.threshold(gray,1,255,cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    x, y, w, h = cv2.boundingRect(cnt)
    crop = img[y:y+h,x:x+w]

    return (imageA,imageB,crop,vis)

'''
    Write Function
Usage:  Writes an image to a file
Input:  writeTo - a path to save he image to. (if the path isn't in this directory put the full path
        img - the image to save (ex: the result of the stitch function)
Output: None (just the saved image)
'''
def write(writeTo,img):
    cv2.imwrite(writeTo,img)

'''
    Show Function
Usage:  Shows the results of the stitch function
Input:  imageA - some stitched image
        imageB - some other stitched image
        stitched - the returned tuple of the stitch function
Output: Shows 4 windows with each image in 1 window
Future: Output into a GUI
'''
def show(imageA, imageB, results, vis):
    cv2.imshow("Image A", imageA)
    cv2.imshow("Image B", imageB)
    cv2.imshow("Keypoint Matches", vis)
    cv2.imshow("Result", results)
    cv2.waitKey(0)

### TEST ###
# def testAll():
    #bryce_01
    # imageA = "images/bryce_left_01.png"
    # imageB = "images/bryce_right_01.png"
    # (imageA,imageB,results,vis) = stitch(imageA,imageB)
    # write("images/tests/bryce_01.png",results)
    # show(imageA,imageB,results,vis)
    #grand_canyon_01
    # imageA = "images/grand_canyon_left_01.png"
    # imageB = "images/grand_canyon_right_01.png"
    # (imageA,imageB,results,vis) = stitch(imageA,imageB)
    # write("images/tests/grand_canyon_01.png",results)
    # show(imageA,imageB,results,vis)
    #scottsdale_01
    # imageA = "images/scottsdale_left_01.png"
    # imageB = "images/scottsdale_right_01.png"
    # (imageA,imageB,results,vis) = stitch(imageA,imageB)
    # write("images/tests/scottsdale_01.png",results)
    # show(imageA,imageB,results,vis)
# testAll()
