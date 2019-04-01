import numpy as np
import matplotlib.pyplot as plt
import PIL.Image
import matplotlib.image as mpimg
import scipy.optimize as so
import cv2
import piexif
import sys
from mpl_toolkits.mplot3d import Axes3D

class Image(object):
    def __init__(self, img): #, imggcp=[], realgcp=[],pose=None):
        #self.imagegcp = imggcp
        #self.realgcp = realgcp
        #self.pose = pose
        
        self.img = plt.imread(img)

        # Image Description
        self.image = piexif.load(img)
        self.h = plt.imread(img).shape[0]
        self.w = plt.imread(img).shape[1]
        self.d = plt.imread(img).shape[2]
        self.f = self.image['Exif'][piexif.ExifIFD.FocalLengthIn35mmFilm]/36*self.w

class camClass(Image):
    def __init__(self): #,pose_guess=None,Kmat=None):
        #self.c = np.array([sensor_x, sensor_y])  # Sensor
        #self.pose_guess = pose_guess
        #self.Kmat = Kmat #camera matrix
        self.images = []
        self.pointCloud = []
    
    def add_images(self,image):
        #image.pose = self.pose_guess  #initialize image with guess
        self.images.append(image)

    def stitch_and_plot(im1,im2):
        I1 = self.images[0]
        I2 = self.images[1]
        h,w,d,f = I1.h, I1.w, I1.d, I1.f

        sift = cv2.xfeatures2d.SIFT_create()
        kp1,des1 = sift.detectAndCompute(I1.img, None)
        kp2,des2 = sift.detectAndCompute(I2.img, None)
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1,des2,k=2)

        # Apply ratio test
        good = []
        for i,(m,n) in enumerate(matches):
            if m.distance < 0.7*n.distance:
                good.append(m)
        u1 = []
        u2 = []
        for m in good:
            u1.append(kp1[m.queryIdx].pt)
            u2.append(kp2[m.trainIdx].pt)

        # General Coordinates
        u1g = np.array(u1)
        u2g = np.array(u2)

        # Make Homogeneous
        u1h = np.c_[u1g,np.ones(u1g.shape[0])]
        u2h = np.c_[u2g,np.ones(u2g.shape[0])]

        # Image Center
        cv = h/2
        cu = w/2

        # Get Camera Coordinates
        K_cam = np.array([[f,0,cu],[0,f,cv],[0,0,1]])
        K_inv = np.linalg.inv(K_cam)
        x1 = u1h @ K_inv.T
        x2 = u2h @ K_inv.T

        # Generate Essential Matrix
        E, inliers = cv2.findEssentialMat(x1[:,:2],x2[:,:2],np.eye(3),method=cv2.RANSAC,threshold=1e-3)
        inliers = inliers.ravel().astype(bool)
        n_in,R,t,_ = cv2.recoverPose(E,x1[inliers,:2],x2[inliers,:2])

        x1 = x1[inliers==True]
        x2 = x2[inliers==True]            
 
        pic = np.zeros(im1.shape*2)
        pic[im1.shape(0):,im1.shape(1):] = im1
        pic[:im1.shape(0),:im1.shape(1)] = im2
	
        fig = plt.figure()
        ax = fig.subplot()
        for i in range(len(x1)):
             ax.plot( (x1(i,0),x1(i,1)), (x2(i,0),x2(i,1)), pic)
        plt.show()

    def genPointCloud(self):
        def triangulate(P0,P1,x1,x2):
            '''X,Y,Z,W of the key points that are found in 2 images
               P0 and P1 are poses, x1 and x2 are SIFT key-points'''
            A = np.array([[P0[2,0]*x1[0] - P0[0,0], P0[2,1]*x1[0] - P0[0,1], P0[2,2]*x1[0] - P0[0,2], P0[2,3]*x1[0] - P0[0,3]],
                         [P0[2,0]*x1[1] - P0[1,0], P0[2,1]*x1[1] - P0[1,1], P0[2,2]*x1[1] - P0[1,2], P0[2,3]*x1[1] - P0[1,3]],
                         [P1[2,0]*x2[0] - P1[0,0], P1[2,1]*x2[0] - P1[0,1], P1[2,2]*x2[0] - P1[0,2], P1[2,3]*x2[0] - P1[0,3]],
                         [P1[2,0]*x2[1] - P1[1,0], P1[2,1]*x2[1] - P1[1,1], P1[2,2]*x2[1] - P1[1,2], P1[2,3]*x2[1] - P1[1,3]]])
            u,s,vt = np.linalg.svd(A)
            return vt[-1]
            
        I1 = self.images[0]
        I2 = self.images[1]
        h,w,d,f = I1.h, I1.w, I1.d, I1.f
	
        sift = cv2.xfeatures2d.SIFT_create()
        kp1,des1 = sift.detectAndCompute(I1.img, None)
        kp2,des2 = sift.detectAndCompute(I2.img, None)
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1,des2,k=2)

	# Apply ratio test
        good = []
        for i,(m,n) in enumerate(matches):
            if m.distance < 0.7*n.distance:
                good.append(m)
        u1 = []
        u2 = []
        for m in good:
            u1.append(kp1[m.queryIdx].pt)
            u2.append(kp2[m.trainIdx].pt)

	# General Coordinates
        u1g = np.array(u1)
        u2g = np.array(u2)

	# Make Homogeneous
        u1h = np.c_[u1g,np.ones(u1g.shape[0])]
        u2h = np.c_[u2g,np.ones(u2g.shape[0])]
		
        # Image Center
        cv = h/2
        cu = w/2

        # Get Camera Coordinates
        K_cam = np.array([[f,0,cu],[0,f,cv],[0,0,1]])
        K_inv = np.linalg.inv(K_cam)
        x1 = u1h @ K_inv.T 
        x2 = u2h @ K_inv.T 

        # Generate Essential Matrix
        E, inliers = cv2.findEssentialMat(x1[:,:2],x2[:,:2],np.eye(3),method=cv2.RANSAC,threshold=1e-3)
        inliers = inliers.ravel().astype(bool) 
        n_in,R,t,_ = cv2.recoverPose(E,x1[inliers,:2],x2[inliers,:2])

        print (x1.shape)

        x1 = x1[inliers==True]
        x2 = x2[inliers==True]

        print (x1.shape)

        # Relative pose between two camperas
        P0 = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])
        P1 = np.hstack((R,t))
        
        # Find X,Y,Z for all SIFT Keypoints
        for i in range(len(x1)):
            self.pointCloud.append(triangulate(P0,P1,x1[i],x2[i]))   #appends to list of points in xyz coordinates

        self.pointCloud = np.array(self.pointCloud)
        self.pointCloud = self.pointCloud.T / self.pointCloud[:,3]
        self.pointCloud = self.pointCloud[:-1,:] 
        print (self.pointCloud.shape)
        
    def plotPointCloud(self):
        #%matplotlib notebook
        fig = plt.figure()
        ax = fig.add_subplot(111,projection='3d')
        ax.plot(*self.pointCloud,'k.')
        plt.show()

if __name__ == "__main__":
  Image1 = Image(sys.argv[1])
  Image2 = Image(sys.argv[2])
  pointCloud = camClass()
  pointCloud.add_images(Image1)
  pointCloud.add_images(Image2)
  pointCloud.genPointCloud()
  pointCloud.plotPointCloud()
