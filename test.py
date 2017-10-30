from panorama import Stitcher
import cv2
#im1,im2,im3,res,vis = s.stitchAll('test_images/halfdome-00.png',
#    'test_images/halfdome-01.png','test_images/halfdome-02.png')
#s.show(im1,im2,im3,res,vis)
stitcher = Stitcher()
perspective = ['test_images/halfdome-00.png',
    'test_images/halfdome-01.png','test_images/halfdome-02.png',
    'test_images/halfdome-03.png','test_images/halfdome-04.png',
    'test_images/halfdome-05.png']
perspectiveReverse = ['test_images/halfdome-05.png',
    'test_images/halfdome-04.png','test_images/halfdome-03.png',
    'test_images/halfdome-02.png','test_images/halfdome-01.png',
    'test_images/halfdome-00.png']
affine = ['test_images/t1.jpg','test_images/t2.jpg','test_images/t3.jpg','test_images/t4.jpg',
          'test_images/t5.jpg','test_images/t6.jpg','test_images/t7.jpg','test_images/t8.jpg',
          'test_images/t9.jpg','test_images/t10.jpg']
diagonalLeftRight = ['test_images/b0.jpg','test_images/b1.jpg','test_images/b2.jpg','test_images/b3.jpg']
diagonalRightLeft = ['test_images/b3.jpg','test_images/b2.jpg','test_images/b1.jpg','test_images/b0.jpg']

two = ['test_images/halfdome-03.png','test_images/halfdome-02.png']

res = stitcher.stitch(affine)
cv2.waitKey(0)
