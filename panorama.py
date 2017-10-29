# import the necessary packages
import numpy as np
import imutils
import cv2

#----------------------------------------------------------------------------
#-----------------------------TO-DO------------------------------------------
#-----------Find out why the offset translation cause a HUGE   --------------
#-----------mis-stiching of the images, the problem is probably -------------
#-----------due to perspective transformation, EROR_FORWARD_CARRYING --------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#SOLUTION: KEEP TRACK OF LINEAR TRANSLATION SEPERATELY

class Stitcher:
	def __init__(self):
		# determine if we are using OpenCV v3.X
		self.isv3 = imutils.is_cv3()
	def stitchLeftToRight(self, images, ratio=0.75, reprojThresh=4.0,
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
			hs.append(self.matchKeypoints(ks[i+1], ks[i],
			fs[i+1], fs[i], ratio, reprojThresh))

		#if homographies does not exist return None
		#(consequtive images does not contain enough matches)
		#CHECK THIS, MIGHT BE WRONG
		if None in hs:
			return None
			

		print(self.calcBoundingRect(imgs,hs))

		while len(imgs)>1:

			#warp the last image with respect to the second to last

			#calculate corner positions of the last image
			w,h = (imgs[-1].shape[1],imgs[-1].shape[0])
			lt = hs[-1][1].dot(np.array([0, 0, 1]))
			lb = hs[-1][1].dot(np.array([0, h,1]))
			rt = hs[-1][1].dot(np.array([w, 0, 1]))
			rb = hs[-1][1].dot(np.array([w, h,1]))

			#If the image is to the left or to the top
			#translate it accordingly
			print(lt,lb)
			xOffset = int(min(lt[0],lb[0]))
			yOffset = int(min(lt[1],rt[1]))

			if xOffset < 0:
				hs[-1][1][0][2] -= xOffset

			if yOffset < 0:
				hs[-1][1][1][2] -= yOffset


			#get the bounding rectangle of the two images
			rect = self.getBoundingRect(0,0,imgs[-2].shape[1],imgs[-2].shape[0],
									xOffset,yOffset,imgs[-1].shape[1],imgs[-1].shape[0]) 

			#warpPerspective BURDAKI OPENCV FUNCTIONINI DEGISTIRDIM HULOOO
			result = self.warp_perspective(imgs[-1], hs[-1][1],
			(rect[2]-rect[0], rect[3]-rect[1]),(xOffset,yOffset))
			cv2.imshow('result'+str(len(imgs)),result)
			cv2.imshow('im1',imgs[-1])
			cv2.imshow('im2',imgs[-2])

			#if the_offset is positive the warpPerspective function takes
			#care of the translation, so no additional off_set is needed
			
			if xOffset>0:
				xOffset = 0
			if yOffset>0:
				yOffset = 0

			"""
			#add borders to the result so that the second image can be added to the first
			result=cv2.copyMakeBorder(result, top=0,
			 bottom=self.negToZero(rect[3]-rect[1]-result.shape[0]),
			 left=0,
			 right=self.negToZero(rect[2]-rect[0]-result.shape[1]),
			 borderType= cv2.BORDER_CONSTANT,
			 value=[0,0,0])
			"""

			#stich them
			result[-(yOffset):-(yOffset)+imgs[-2].shape[0],
			    -(xOffset):-(xOffset)+imgs[-2].shape[1]] = imgs[-2]


			#account for the offset we added to the images
			globalYOffset += xOffset
			globalXOffset += yOffset 
			
			for i in range(len(hs)):
				hs[i][1][0][2] += xOffset
				hs[i][1][1][2] += yOffset

			#for i in range(len(hs)):
			#	hs[i][1][0][2] += -hs[i][1][0][0]*xOffset - hs[i][1][0][1]*yOffset
			#	hs[i][1][1][2] += -hs[i][1][1][0]*xOffset - hs[i][1][1][1]*yOffset
				

			#remove the last elements of all the lists
			ks.pop()
			fs.pop()
			hs.pop()
			imgs.pop()
			imgs.pop()
			imgs.append(result)

		return result

	def stitch2(self, images, ratio=0.75, reprojThresh=4.0,
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
			hs.append(self.matchKeypoints(ks[i+1], ks[i],
			fs[i+1], fs[i], ratio, reprojThresh))
		#first homography matrix
		homography = np.identity(3)
		#if homographies does not exist return None
		#(consequtive images does not contain enough matches)
		#CHECK THIS, MIGHT BE WRONG
		#if all(hs[1]):
		#	return None
		

		print(self.calcBoundingRect(imgs,hs))

		lt,rb = self.calcBoundingRect(imgs,hs)

		xOffset, yOffset = int(lt[0]),int(lt[1])
		w,h = int(rb[0])-xOffset, int(rb[1])-yOffset
		print(xOffset,yOffset,w,h)

		main = cv2.warpPerspective(imgs[0], homography,
			(w, h))

		for i in range(len(imgs)):

			if xOffset < 0:
				hs[0][1][0][2] += -xOffset

			if yOffset < 0:
				hs[0][1][1][2] += -yOffset


			#warpPerspective
			print(homography)
			result = cv2.warpPerspective(imgs[i], homography,
			(w, h))
			cv2.imshow('result'+str(i),result)
			cv2.imshow('im1',imgs[-1])
			cv2.imshow('im2',imgs[-2])

			grayR = cv2.cvtColor(result,cv2.COLOR_BGR2GRAY)
			retR, maskR = cv2.threshold(grayR, 10, 255, cv2.THRESH_BINARY)

			grayM = cv2.cvtColor(main,cv2.COLOR_BGR2GRAY)
			retM, maskM = cv2.threshold(grayM, 10, 255, cv2.THRESH_BINARY_INV)
			cv2.imshow('maskM'+str(i),maskM)

			maskedR = cv2.bitwise_and(main,maskM)


			#masked = cv2.bitwise_and(main,result,mask = mask)
			cv2.imshow('maskedR'+str(i),maskR)
			#cv2.imshow('maskM'+str(i),maskM)



			#stich them
			main = cv2.bitwise_or(main,result)

			if i < len(hs):
				homography = homography.dot(hs[i][1])
			#for i in range(len(hs)):
			#	hs[i][1][0][2] += -hs[i][1][0][0]*xOffset - hs[i][1][0][1]*yOffset
			#	hs[i][1][1][2] += -hs[i][1][1][0]*xOffset - hs[i][1][1][1]*yOffset
				

			#remove the last elements of all the lists
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
				h = h.dot(hs[i][1])
		lt,rb = ((min(points,key = lambda t: t[0])[0],min(points,key = lambda t: t[1])[1]),
				(max(points,key = lambda t: t[0])[0],max(points,key = lambda t: t[1])[1]))
		return (lt,rb)

	def warp_perspective(self,src, M, (width, height), (origin_x, origin_y),
	                     flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
	                     borderValue=0, dst=None):
	    """
	    Implementation in Python using base code from
	    http://stackoverflow.com/questions/4279008/specify-an-origin-to-warpperspective-function-in-opencv-2-x

	    Note there is an issue with linear interpolation.
	    """
	    B_SIZE = 32

	    if dst == None:
	        dst = np.zeros((height, width, 3), dtype=src.dtype)

	    # Set interpolation mode.
	    interpolation = flags & cv2.INTER_MAX
	    if interpolation == cv2.INTER_AREA:
	        raise Exception('Area interpolation is not supported!')

	    # Prepare matrix.    
	    M = M.astype(np.float64)
	    if not(flags & cv2.WARP_INVERSE_MAP):
	        M = cv2.invert(M)[1]
	    M = M.flatten()

	    x_dst = y_dst = 0
	    for y in xrange(-origin_y, height, B_SIZE):
	        for x in xrange(-origin_x, width, B_SIZE):

	            print (x, y)

	            # Block dimensions.
	            bw = min(B_SIZE, width - x_dst)
	            bh = min(B_SIZE, height - y_dst)

	            # To avoid dimension errors.
	            if bw <= 0 or bh <= 0:
	                break

	            # View of the destination array.
	            dpart = dst[y_dst:y_dst+bh, x_dst:x_dst+bw]

	            # Original code used view of array here, but we're using numpy array's.
	            XY = np.zeros((bh, bw, 2), dtype=np.int16)
	            A = np.zeros((bh, bw), dtype=np.uint16)

	            for y1 in xrange(bh):
	                X0 = M[0]*x + M[1]*(y + y1) + M[2]
	                Y0 = M[3]*x + M[4]*(y + y1) + M[5]
	                W0 = M[6]*x + M[7]*(y + y1) + M[8]

	                if interpolation == cv2.INTER_NEAREST:
	                    for x1 in xrange(bw):
	                        W = np.float64(W0 + M[6]*x1);
	                        if W != 0:
	                            W = np.float64(1.0)/W

	                        X = np.int32((X0 + M[0]*x1)*W)
	                        Y = np.int32((Y0 + M[3]*x1)*W)
	                        XY[y1, x1][0] = np.int16(X)
	                        XY[y1, x1][1] = np.int16(Y)
	                else:
	                    for x1 in xrange(bw):
	                        W = np.float64(W0 + M[6]*x1);
	                        if W != 0:
	                            W = cv2.INTER_TAB_SIZE/W

	                        X = np.int32((X0 + M[0]*x1)*W)
	                        Y = np.int32((Y0 + M[3]*x1)*W)
	                        XY[y1, x1][0] = np.int16((X >> cv2.INTER_BITS) + origin_x)
	                        XY[y1, x1][1] = np.int16((Y >> cv2.INTER_BITS) + origin_y)
	                        A[y1, x1] = np.int16(((Y & (cv2.INTER_TAB_SIZE-1))*cv2.INTER_TAB_SIZE + (X & (cv2.INTER_TAB_SIZE-1))))

	            if interpolation == cv2.INTER_NEAREST:
	                cv2.remap(src, XY, None, interpolation, dst=dpart,
	                          borderMode=borderMode, borderValue=borderValue)
	            else:
	                cv2.remap(src, XY, A, interpolation, dst=dpart,
	                          borderMode=borderMode, borderValue=borderValue)

	            x_dst += B_SIZE
	        x_dst = 0
	        y_dst += B_SIZE


