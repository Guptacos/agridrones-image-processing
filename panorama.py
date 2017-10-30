# import the necessary packages
import numpy as np
import imutils
import cv2
from copy import*

#TO-DO: FIND THE REASON WHY IT DOESNT WORK FOR PERSPECTIVE TRANFORMATION (CHECK BOUNDING RECT)
#       FIX COMBINING ALGORITHM

class Stitcher:
	def __init__(self):
		# determine if we are using OpenCV v3.X
		self.isv3 = imutils.is_cv3()

	#DOESNT WORK WITH PERSPECTIVE
	def stitch(self, images, ratio=0.75, reprojThresh=4.0,
		showMatches=False,resize = 400):
		#copy of images
		imgs = []
		for image in images:
			img = cv2.imread(image)
			if resize is not None:
				img = imutils.resize(img, width=resize)
			imgs.append(img)
		globalXOffset = 0
		globalYOffset = 0
		#keypoints
		ks = []
		#features
		fs = []
		#homographies, in the form of i+1 respect to i
		hs = []
		#find keypoints and features
		for image in imgs:
			kp,ft = self.detectAndDescribe(image)
			ks.append(kp)
			fs.append(ft)

		#calculate homographies
		for i in range(len(imgs)-1):
			hs.append((self.matchKeypoints(ks[i+1], ks[i],
			fs[i+1], fs[i], ratio, reprojThresh))[1])

		lt,rb = self.calcBoundingRect(imgs,hs)

		xOffset, yOffset = int(lt[0]),int(lt[1])
		w,h = int(rb[0])-xOffset, int(rb[1])-yOffset

		#create empty image of appropriate size		
		main = np.zeros((h,w,3), np.uint8)

		#add identity matrix as the first homography matrix
		hs.insert(0,np.identity(3))
		
		#instantiate base homography matrix
		homography = np.identity(3)

		for i in range(len(imgs)):
			#update base homography matrix
			homography = homography.dot(hs[i])

			#create a copy of the base homography matrix to
			#translate it without affecting the other homographies
			hTranslated = deepcopy(homography)
			hTranslated[0][2] += -xOffset
			hTranslated[1][2] += -yOffset

			#warpPerspective
			result = cv2.warpPerspective(imgs[i], hTranslated,
			(w, h))
			#cv2.imshow('result'+str(i),result)
			#cv2.imshow('im1',imgs[-1])
			#cv2.imshow('im2',imgs[-2])

			#stich them with masking
			anded = cv2.bitwise_and(result,main)
			gray = cv2.cvtColor(anded, cv2.COLOR_BGR2GRAY)
			ret,binary = cv2.threshold(gray,1,255,cv2.THRESH_BINARY_INV)
			binary = cv2.cvtColor(binary,cv2.COLOR_GRAY2RGB)
			maskedMain = cv2.bitwise_and(main,binary)
			main = cv2.bitwise_or(maskedMain,result)

		cv2.imshow('main',main)
		return result


	def detectAndDescribe(self, image):
		# convert the image to grayscale
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		# check to see if we are using OpenCV 3.X
		if self.isv3:
			# detect and extract features from the image
			descriptor = cv2.xfeatures2d.SIFT_create()
			(kps, features) = descriptor.detectAndCompute(image, None)

		# otherwise, we are using OpenCV 2.4.X
		else:
			# detect keypoints in the image
			detector = cv2.FeatureDetector_create("SIFT")
			kps = detector.detect(gray)

			# extract features from the image
			extractor = cv2.DescriptorExtractor_create("SIFT")
			(kps, features) = extractor.compute(gray, kps)

		# convert the keypoints from KeyPoint objects to NumPy
		# arrays
		kps = np.float32([kp.pt for kp in kps])

		# return a tuple of keypoints and features
		return (kps, features)

	def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB,
		ratio, reprojThresh):
		# compute the raw matches and initialize the list of actual
		# matches
		matcher = cv2.DescriptorMatcher_create("BruteForce")
		rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
		matches = []

		# loop over the raw matches
		for m in rawMatches:
			# ensure the distance is within a certain ratio of each
			# other (i.e. Lowe's ratio test)
			if len(m) == 2 and m[0].distance < m[1].distance * ratio:
				matches.append((m[0].trainIdx, m[0].queryIdx))

		# computing a homography requires at least 4 matches
		if len(matches) > 4:
			# construct the two sets of points
			ptsA = np.float32([kpsA[i] for (_, i) in matches])
			ptsB = np.float32([kpsB[i] for (i, _) in matches])

			# compute the homography between the two sets of points
			(H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
				reprojThresh)

			# return the matches along with the homograpy matrix
			# and status of each matched point
			return (matches, H, status)

		# otherwise, no homograpy could be computed
		return None

	def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status):
		# initialize the output visualization image
		(hA, wA) = imageA.shape[:2]
		(hB, wB) = imageB.shape[:2]
		vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
		vis[0:hA, 0:wA] = imageA
		vis[0:hB, wA:] = imageB

		# loop over the matches
		for ((trainIdx, queryIdx), s) in zip(matches, status):
			# only process the match if the keypoint was successfully
			# matched
			if s == 1:
				# draw the match
				ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
				ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
				cv2.line(vis, ptA, ptB, (0, 255, 0), 1)

		# return the visualization
		return vis


	def getBoundingRect(self,x0,y0,w0,h0,x1,y1,w1,h1):
		return (min(x0,x1),min(y0,y1),max(x0+w0,x1+w1),max(y0+h0,y1+h1))

	def negToZero(self,number):
		if number < 0:
			return 0
		return number

	def calcBoundingRect(self,imgs,hs):
		points = []
		h = np.identity(3)
		for i in range(len(imgs)):
			width,height = (imgs[i].shape[1],imgs[i].shape[0])
			lt = h.dot(np.array([0, 0, 1]))
			lb = h.dot(np.array([0, height,1]))
			rt = h.dot(np.array([width, 0, 1]))
			rb = h.dot(np.array([width, height,1]))
			points.extend([lt,lb,rt,rb])
			if i < len(imgs)-1:
				h = h.dot(hs[i])
		lt,rb = ((min(points,key = lambda t: t[0])[0],min(points,key = lambda t: t[1])[1]),
				(max(points,key = lambda t: t[0])[0],max(points,key = lambda t: t[1])[1]))
		return (lt,rb)
