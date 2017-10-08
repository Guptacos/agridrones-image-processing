# import the necessary packages
import numpy as np
import imutils
import cv2

class Stitcher:
	def __init__(self):
		# determine if we are using OpenCV v3.X
		self.isv3 = imutils.is_cv3()

	def stitch(self, images, ratio=0.75, reprojThresh=4.0,
		showMatches=False,resize = 400):
		#copy of images
		imgs = []
		for image in images:
			img = cv2.imread(image)
			img = imutils.resize(img, width=resize)
			imgs.append(img)
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
			hs.append(self.matchKeypoints(ks[i+1], ks[i],
			fs[i+1], fs[i], ratio, reprojThresh))

		#if homographies does not exist return None
		#(consequtive images does not contain enough matches)
		#CHECK THIS, MIGHT BE WRONG
		if None in hs:
			return None
		while len(imgs)>1:
			print(hs[-1][1])
			#warp the last image with respect to the second to last
			result = cv2.warpPerspective(imgs[-1], hs[-1][1],
			(imgs[-1].shape[1] + imgs[-2].shape[1], imgs[-1].shape[0]))

			#stich them (need better stiching algorithm)
			result[0:imgs[-2].shape[0], 0:imgs[-2].shape[1]] = imgs[-2]

			#remove the last elements of all the lists
			ks.pop()
			fs.pop()
			hs.pop()
			imgs.pop()
			imgs.pop()
			imgs.append(result)

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