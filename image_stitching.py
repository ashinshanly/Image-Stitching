import matplotlib
import matplotlib.pyplot as plt
import imageio
import imutils
import random
import cv2
import numpy as np

#Function to convert a matrix to an image.
def convert_to_image(matrix):
    V,H,C = matrix.shape
    #Initialize image with all zero values.
    image = np.zeros((H,V,C), dtype='uint8')
    for i in range(matrix.shape[0]):
        image[:,i] = matrix[i]
    return image

#Function to convert an image to a matrix.
def convert_to_matrix(image):
    H,V,C = image.shape
    #Initialize matrix with all zeroes.
    matrix = np.zeros((V,H,C), dtype='int')
    for i in range(image.shape[0]):
        matrix[:,i] = image[i]    
    return matrix

#Function to find keypoints and features.
def findAndDescribeFeatures(image):

    # Convert to grayscale.
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #Create ORB object
    obj = cv2.ORB_create(nfeatures=3000)
    # Find keypoints and features.
    keypoints, features = obj.detectAndCompute(grayImage, None)
    #Convert features to float.
    features = np.float32(features)
    return keypoints, features

#Function to match the features between two images.
def matchFeatures(featuresA, featuresB, ratio=0.75):

    #Here FLANN (Fast Library for Approximate Nearest Neighbors) is used as it contains a collection of algorithms optimized for fast nearest neighbor search in large datasets and for high dimensional features, and works faster than BFMatcher for large datasets.
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    featureMatcher = cv2.FlannBasedMatcher(index_params, search_params)
    matches = featureMatcher.knnMatch(featuresA, featuresB, k=2) #2-NN matching is used here.
    close_matches = [] #Contains closer matches according to ratio given.
    for m, n in matches:
        if m.distance < ratio * n.distance:
            close_matches.append(m)
    if len(close_matches) > 4: #Atleast 4 points are needed.
        return close_matches

#Function to find distance error.
def Distance(correspondence, H):
    p1 = np.transpose(np.matrix([correspondence[0].item(0), correspondence[0].item(1), 1]))
    estimatep2 = np.dot(H, p1)
    estimatep2 = (1/estimatep2.item(2))*estimatep2

    p2 = np.transpose(np.matrix([correspondence[0].item(2), correspondence[0].item(3), 1]))
    error = p2 - estimatep2
    return np.linalg.norm(error)

#Function to calculate homography matrix.
def findHomography(correspondences):
    assemble = []
    for correspondence in correspondences:
        point1 = np.matrix([correspondence.item(0), correspondence.item(1), 1])
        point2 = np.matrix([correspondence.item(2), correspondence.item(3), 1])

        assembleR2 = [0, 0, 0, -point2.item(2) * point1.item(0), -point2.item(2) * point1.item(1), -point2.item(2) * point1.item(2),
              point2.item(1) * point1.item(0), point2.item(1) * point1.item(1), point2.item(1) * point1.item(2)]
        assembleR1 = [-point2.item(2) * point1.item(0), -point2.item(2) * point1.item(1), -point2.item(2) * point1.item(2), 0, 0, 0,
              point2.item(0) * point1.item(0), point2.item(0) * point1.item(1), point2.item(0) * point1.item(2)]
        #Insert to assemble list.
        assemble.append(assembleR1)
        assemble.append(assembleR2)

    #Convert to matrix format.
    assembleMat = np.matrix(assemble) #Assemble Matrix.
    #Apply SVD.
    u, s, v = np.linalg.svd(assembleMat)
    #Reshape into a 3 by 3 matrix.
    h = np.reshape(v[8], (3, 3))
    #Normalize.
    h = (1/h.item(8)) * h
    return h

#Ransac Algorithm
def execute_ransac(correspondencesMatrix, thresh):

    Inliers_thresh = []
    homgraphyMatrix = None
    #Ransac will loop 1000 times here, but will break if it finds a
    for i in range(1000):  
        #Find Random points(4) to estimate a homography matrix.
        point1 = correspondencesMatrix[random.randint(0, len(correspondencesMatrix)-1)]  
        point2 = correspondencesMatrix[random.randint(0, len(correspondencesMatrix)-1)]
        fourRandomPoints = np.vstack((point1, point2))

        point3 = correspondencesMatrix[random.randint(0, len(correspondencesMatrix)-1)]
        fourRandomPoints = np.vstack((fourRandomPoints, point3))
        
        point4 = correspondencesMatrix[random.randint(0, len(correspondencesMatrix)-1)]
        fourRandomPoints = np.vstack((fourRandomPoints, point4))
        #Calculate homography corresponding to those points.
        H = findHomography(fourRandomPoints) 
        inliers = []

        for i in range(len(correspondencesMatrix)):
            #Calculate geometric distance between points. 
            distance = Distance(correspondencesMatrix[i], H) 
            #Choose only close ones.
            if distance < 5:  
                inliers.append(correspondencesMatrix[i])
        
        #Check if threshold of number of inliers is attained.
        if len(inliers) > len(Inliers_thresh):
            homgraphyMatrix = H
            Inliers_thresh = inliers
            
        #Stop if minimum percentage of image points that the current best homography estimation accounts for is achieved. 
        if len(Inliers_thresh) > (len(correspondencesMatrix)*thresh):
            break

    return homgraphyMatrix

#Implementation of Warping.
def perspectiveWarp(image, H, dimensions):
    #Convert src_image to matrix first.
    matrix = convert_to_matrix(image)
    R, C = dimensions
    source = np.zeros((R,C,matrix.shape[2]))
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            d = np.dot(H, [i,j,1])
            d = np.ravel(d)
            i_hat,j_hat,_ = (d / d[2] + 0.5).astype(int)
            if i_hat >= 0 and i_hat < R:
                if j_hat >= 0 and j_hat < C:
                    source[i_hat,j_hat] = matrix[i,j]
    #Convert src_image matrix back to image format.
    return convert_to_image(source)

#Mask for blending to obtain a smooth transient
def blendingMask(height, width, barrier, smoothing_window, is_left=True): 
    assert barrier < width
    mask = np.zeros((height, width))

    offset = int(smoothing_window / 2)
    """
    if is_left:
        mask[:, barrier - offset : barrier + offset + 1] = np.tile(np.linspace(1, 0, 2 * offset + 1).T, (height, 1))
        mask[:, : barrier - offset] = 1
    else:
        mask[:, barrier - offset : barrier + offset + 1] = np.tile(np.linspace(0, 1, 2 * offset + 1).T, (height, 1))
        mask[:, barrier + offset :] = 1
    """
    try:
        if is_left:
            mask[:, barrier - offset : barrier + offset + 1] = np.tile(np.linspace(1, 0, 2 * offset + 1).T, (height, 1))
            mask[:, : barrier - offset] = 1
        else:
            mask[:, barrier - offset : barrier + offset + 1] = np.tile(np.linspace(0, 1, 2 * offset + 1).T, (height, 1))
            mask[:, barrier + offset :] = 1

    except BaseException:
        if is_left:
            mask[:, barrier - offset : barrier + offset + 1] = np.tile(np.linspace(1, 0, 2 * offset).T, (height, 1))
            mask[:, : barrier - offset] = 1
        else:
            mask[:, barrier - offset : barrier + offset + 1] = np.tile(np.linspace(0, 1, 2 * offset).T, (height, 1))
            mask[:, barrier + offset :] = 1

    return cv2.merge([mask, mask, mask])

#Own implementation of homography estimation
#(src_image is to be wrapped using homography).
def generateMyHomography(src_img, dst_img): 
        #Create ORB object.
        obj = cv2.ORB_create(nfeatures=3000)
        #Convert src_img to grayscale.
        graySRC = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
        kpSRC, featuresSRC = obj.detectAndCompute(graySRC, None)
        #Convert dst_img to grayscale.
        grayDST = cv2.cvtColor(dst_img, cv2.COLOR_BGR2GRAY)
        kpDST, featuresDST = obj.detectAndCompute(grayDST, None)
        #featuresSRC = np.float32(featuresSRC)
        #featuresDST = np.float32(featuresDST)
        ratio = 0.75

        #Norm hamming is preferred for ORB.
        bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        allMatches = bf_matcher.knnMatch(featuresSRC, featuresDST, 2)
        matches = []  
        #matches[] contain all matches which are within a specific ratio of distance apart.
        for l, r in allMatches:
            if l.distance < r.distance * ratio:
                matches.append(l)

        keypoints = [kpSRC,kpDST] #Holds all keypoints.
        correspondences_list = [] #Holds all the correspondences.
        for match in matches:
            (x1, y1) = keypoints[0][match.queryIdx].pt
            (x2, y2) = keypoints[1][match.trainIdx].pt
            correspondences_list.append([x1, y1, x2, y2])
        #Convert to matrix 
        correspondencesMatrix = np.matrix(correspondences_list)

        #Call Ransac algorithm
        homographyMatrix = execute_ransac(correspondencesMatrix, thresh=0.4)
        return homographyMatrix

#Inbuilt homography estimator function
def generateHomography(src_img, dst_img, ransacRep=5.0):

    src_kp, src_features = findAndDescribeFeatures(src_img)
    dst_kp, dst_features = findAndDescribeFeatures(dst_img)

    good = matchFeatures(src_features, dst_features)  #Match the features.

    src_points = np.float32([src_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_points = np.float32([dst_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC, ransacRep)
    return H

#Creates smooth transient and merges two images. 
def blendAndMerge(dst_img_hat, src_img_warped, width_dst, position):
    height, width, _ = dst_img_hat.shape
    #Calculate masks for blending in both the positions of stitching(left and right).
    mask1 = blendingMask(height, width, width_dst-int(int(width_dst/8)/2), smoothing_window=int(width_dst/8), is_left=True)
    mask2 = blendingMask(height, width, width_dst-int(int(width_dst/8)/2), smoothing_window=int(width_dst/8), is_left=False)
    if position == "left":
        #Flip both images along vertical axis.
        dst_img_hat = cv2.flip(dst_img_hat, 1)
        src_img_warped = cv2.flip(src_img_warped, 1)
        #Multiply both flipped images by the respective masks to obtain a smooth transient, i.e even out pixel intensities etc.
        dst_img_hat = dst_img_hat * mask1
        src_img_warped = src_img_warped * mask2
        pano = src_img_warped + dst_img_hat
        pano = cv2.flip(pano, 1)
    else:
        #Multiply both flipped images by the respective masks without flipping since position='right'.
        dst_img_hat = dst_img_hat * mask1
        src_img_warped = src_img_warped * mask2
        pano = src_img_warped + dst_img_hat
    #Return blended and merged panorama of 2 images.
    return pano

#Crop out black regions.
def crop(panorama, dst_height, boundaries):
    #Boundaries contain the corner points of both the images.
    boundaries = boundaries.astype(int)

    # find maximum and minimum of x,y coordinates to estimate the boundaries of merged image.
    [xmin, ymin] = np.int32(boundaries.min(axis=0).ravel() - 0.5)
    min_coordinates = [-xmin, -ymin]

    #Eliminate all extra regions that fall out of the corners of the final panorama.
    if boundaries[0][0][0] < 0:
        e = abs(-boundaries[1][0][0] + boundaries[0][0][0])
        panorama = panorama[min_coordinates[1] : dst_height + min_coordinates[1], e:, :]
    else:
        if boundaries[2][0][0] < boundaries[3][0][0]:
            panorama = panorama[min_coordinates[1] : dst_height + min_coordinates[1], 0 : boundaries[2][0][0], :]
        else:
            panorama = panorama[min_coordinates[1] : dst_height + min_coordinates[1], 0 : boundaries[3][0][0], :]
    return panorama

#Function to stitch 2 images.
def stitch(src_img, dst_img, opt):

    #Own homography implementation.
    if opt == 'my':
        H = generateMyHomography(src_img, dst_img)
    #Inbuilt Homography.
    elif opt == 'inbuilt':
        H = generateHomography(src_img, dst_img) 
    else:
        print("Check option!")
    
    height_src, width_src = src_img.shape[:2]
    height_dst, width_dst = dst_img.shape[:2]
    #height_src=height_src+1
    #width_dst=width_dst+1

    #Corners of both images
    corners_SRC = np.float32([[0, 0], [0, height_src], [width_src, height_src], [width_src, 0]]).reshape(-1, 1, 2)
    corners_DST = np.float32([[0, 0], [0, height_dst], [width_dst, height_dst], [width_dst, 0]]).reshape(-1, 1, 2)

    try:
        #corners_SRC_transformed will contain the corners of src_image after undergoing perspective transform according to H.
        corners_SRC_transformed = cv2.perspectiveTransform(corners_SRC, H)
        corners = np.concatenate((corners_SRC_transformed, corners_DST), axis=0) #Contains transformed src_image coordinates and corners of original dst_image.
        [xmin, ymin] = np.int64(corners.min(axis=0).ravel() - 0.5) #Minimum of corner coordinates.
        [_, ymax] = np.int64(corners.max(axis=0).ravel() + 0.5) #Maximum of corner coordinates.
        minimum_corner = [-xmin, -ymin]

        #If top left corner is negative, stitch the warped src_image to the left side of dst_image.
        #side indicates the position of stitching of the warped src_image to the dst_image.
        if corners[0][0][0] < 0:
            side = "left" 
            width_pano = width_dst + minimum_corner[0]
        else:
            width_pano = 2*int(corners_SRC_transformed[3][0][0])
            side = "right"
        height_pano = ymax - ymin
        Ht = np.array([[1, 0, minimum_corner[0]], [0, 1, minimum_corner[1]], [0, 0, 1]])

        if opt == "inbuilt":
            #Inbuilt Warp
            src_img_warped = cv2.warpPerspective(src_img, Ht.dot(H), (width_pano, height_pano))

        elif opt=="my":
            #Warp Implementation
            #src_img_warped = perspectiveWarp(src_img, H, (width_pano, height_pano))
            src_img_warped = cv2.warpPerspective(src_img, Ht.dot(H), (width_pano, height_pano))

        #Create a bigger image dst_img_hat with the dimensions of the final panorama. 
        dst_img_hat = np.zeros((height_pano, width_pano, 3))

        #If side is 'left', insert dst_image aligned to the right side of the final panorama.
        if side == "left":
            dst_img_hat[minimum_corner[1] : height_src + minimum_corner[1], minimum_corner[0] : dst_img.shape[1] + minimum_corner[0]] = dst_img
        #If side is 'right', insert dst_image aligned to the left side of the final panorama.
        else:
            dst_img_hat[minimum_corner[1] : height_src + minimum_corner[1], 0:dst_img.shape[1]] = dst_img

        # blending panorama
        pano = blendAndMerge(dst_img_hat, src_img_warped, dst_img.shape[1], side)

        # cropping black region
        pano = crop(pano, height_dst, corners)
        return pano

    except BaseException:
        raise Exception("Try again!")

#Function for stitching multiple images.
#opt argument contains information as to use inbuilt homography or not.
def stitchMultiple(list_images, opt):
    #Find center image.
    center = int(len(list_images) / 2 + 0.5)
    #Separate into 2 subarrays.
    left_subarray = list_images[:center]
    right_subarray = list_images[center - 1 :]
    #Reverse to get the images in order when popping.
    right_subarray.reverse()

    #Perform leftward stitching
    while len(left_subarray) > 1:
        dst_img = left_subarray.pop()
        src_img = left_subarray.pop()
        left_combined = stitch(src_img, dst_img, opt)
        left_combined = left_combined.astype("uint8")
        left_subarray.append(left_combined)
    #Perform rightward stitching
    while len(right_subarray) > 1:
        dst_img = right_subarray.pop()
        src_img = right_subarray.pop()
        right_combined = stitch(src_img, dst_img, opt)
        right_combined = right_combined.astype("uint8")
        right_subarray.append(right_combined)

    #Make src_image(to be warped) as left_combined if its width is smaller than that of right_combined's.
    if right_combined.shape[1] >= left_combined.shape[1]:
        combined = stitch(left_combined, right_combined, opt)
    #If left_combined has more width, then make that as the dst_image.
    else:
        combined = stitch(right_combined, left_combined, opt)
    
    return combined #combine will contain the final panorama.

#---------->
#1st Folder.

images = ['STB_0032.JPG','STC_0033.JPG','STD_0034.JPG','STE_0035.JPG','STF_0036.JPG']
list_of_images = []

for i in images:
  list_of_images.append(cv2.imread(i))

#opt='my' indicates the usage of own implementation of homography estimation.
panorama=stitchMultiple(list_of_images, opt='my')
plt.figure(figsize=(20,20))
#Reduce range to properly print using matplotlib.
panorama = np.array(panorama,dtype=float)/float(255)
panorama = panorama[:,:,::-1]
plt.imshow(panorama)

images = ['STB_0032.JPG','STC_0033.JPG','STD_0034.JPG','STE_0035.JPG','STF_0036.JPG']
list_of_images = []

for i in images:
  list_of_images.append(cv2.imread(i))

#opt='inbuilt' indicates the usage of inbuilt homography estimation.  
panorama=stitchMultiple(list_of_images, opt='inbuilt')
plt.figure(figsize=(20,20))
panorama = np.array(panorama,dtype=float)/float(255)
panorama = panorama[:,:,::-1]
plt.imshow(panorama)

#---------->
#2nd Folder.

images = ['2_2.JPG','2_3.JPG','2_4.JPG','2_5.JPG']
list_of_images = []

for i in images:
  list_of_images.append(cv2.imread(i))
  
panorama=stitchMultiple(list_of_images, opt='inbuilt')
plt.figure(figsize=(20,20))
panorama = np.array(panorama,dtype=float)/float(255)
panorama = panorama[:,:,::-1]
plt.imshow(panorama)


images = ['2_2.JPG','2_3.JPG','2_4.JPG','2_5.JPG']
list_of_images = []

for i in images:
  list_of_images.append(cv2.imread(i))

panorama=stitchMultiple(list_of_images, opt='my')
plt.figure(figsize=(20,20))
panorama = np.array(panorama,dtype=float)/float(255)
panorama = panorama[:,:,::-1]
plt.imshow(panorama)

#---------->
#3rd Folder.

#images = ['3_1.JPG','3_2.JPG','3_3.JPG','3_4.JPG','3_5.JPG']
images = ['3_3.JPG','3_4.JPG','3_5.JPG']
list_of_images = []

for i in images:
  list_of_images.append(cv2.imread(i))
  
panorama=stitchMultiple(list_of_images, opt='my')
plt.figure(figsize=(20,20))
panorama = np.array(panorama,dtype=float)/float(255)
panorama = panorama[:,:,::-1]
plt.imshow(panorama)


#images = ['3_1.JPG','3_2.JPG','3_3.JPG','3_4.JPG']
images = ['3_3.JPG','3_4.JPG','3_5.JPG']

list_of_images = []

for i in images:
  list_of_images.append(cv2.imread(i))
  
panorama=stitchMultiple(list_of_images, opt='inbuilt')
plt.figure(figsize=(20,20))
panorama = np.array(panorama,dtype=float)/float(255)
panorama = panorama[:,:,::-1]
plt.imshow(panorama)

#---------->
#4th Folder.

images = ['DSC02931.JPG','DSC02932.JPG','DSC02933.JPG','DSC02934.JPG']
list_of_images = []

for i in images:
  list_of_images.append(cv2.imread(i))
  
panorama=stitchMultiple(list_of_images, opt='my')
plt.figure(figsize=(20,20))
panorama = np.array(panorama,dtype=float)/float(255)
panorama = panorama[:,:,::-1]
plt.imshow(panorama)


images = ['DSC02931.JPG','DSC02932.JPG','DSC02933.JPG','DSC02934.JPG']
list_of_images = []

for i in images:
  list_of_images.append(cv2.imread(i))
  
panorama=stitchMultiple(list_of_images, opt='inbuilt')
plt.figure(figsize=(20,20))
panorama = np.array(panorama,dtype=float)/float(255)
panorama = panorama[:,:,::-1]
plt.imshow(panorama)

#---------->
#5th Folder

images = ['DSC03002.JPG','DSC03003.JPG','DSC03004.JPG','DSC03005.JPG','DSC03006.JPG']
list_of_images = []

for i in images:
  list_of_images.append(cv2.imread(i))
  
panorama=stitchMultiple(list_of_images, opt='my')
plt.figure(figsize=(20,20))
panorama = np.array(panorama,dtype=float)/float(255)
panorama = panorama[:,:,::-1]
plt.imshow(panorama)


images = ['DSC03002.JPG','DSC03003.JPG','DSC03004.JPG','DSC03005.JPG','DSC03006.JPG']
list_of_images = []

for i in images:
  list_of_images.append(cv2.imread(i))
  
panorama=stitchMultiple(list_of_images, opt='inbuilt')
plt.figure(figsize=(20,20))
panorama = np.array(panorama,dtype=float)/float(255)
panorama = panorama[:,:,::-1]
plt.imshow(panorama)

#---------->
#6th Folder.

#images = ['1_2.JPG','1_3.JPG','1_4.JPG','1_5.JPG']
images = ['1_5.JPG','1_4.JPG','1_3.JPG','1_2.JPG']

list_of_images = []
for i in images:
  list_of_images.append(cv2.imread(i))
  
panorama=stitchMultiple(list_of_images, opt='my')
plt.figure(figsize=(20,20))
panorama = np.array(panorama,dtype=float)/float(255)
panorama = panorama[:,:,::-1]
plt.imshow(panorama)


images = ['1_5.JPG','1_4.JPG','1_3.JPG','1_2.JPG']
list_of_images = []

for i in images:
  list_of_images.append(cv2.imread(i))
  
panorama=stitchMultiple(list_of_images, opt='inbuilt')
plt.figure(figsize=(20,20))
panorama = np.array(panorama,dtype=float)/float(255)
panorama = panorama[:,:,::-1]
plt.imshow(panorama)