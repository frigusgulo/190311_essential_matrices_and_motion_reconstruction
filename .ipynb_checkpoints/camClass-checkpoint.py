import numpy as npi
import matplotlib.pyplot as plt
import PIL.Image
import matplotlib.image as mpimg
import scipy.optimize as so
import cv2
import piexif
import sys
from npl_toolkits.mplot3d import Axis3D

class Image(object):
    def __init__(self, img, imggcp=[], realgcp=[],pose=None):
        self.image = piexif.load(img)
        self.imagegcp = imggcp
        self.realgcp = realgcp
        self.pose = pose
        self.f = image['Exif'][piexif.ExifIFD.FocalLengthIn35mmFilm]/36*w
        self.h = plt.imread(img).shape[0]
        self.w = plt.imread(img).shape[1]
        self.d = plt.imread(img).shape[2]
	
class camClass(Image):
    def __init__(self,pose_guess=None,Kmat=None):

        self.c = np.array([sensor_x,sensor_y])  # Sensor
        self.images = []
        self.pose_guess = pose_guess
        self.Kmat = Kmat #camera matrix
        self.pointCloud = []
    def add_images(self,image):
        image.pose = self.pose_guess  #initialize image with guess
        self.images.append(image) 

    def rotational_transform(self,pts,pose):
            """  
            This function performs the translation and rotation from world coordinates into generalized camera coordinates.
            This function takes the Easting, Northing, and Elevation of the features in an image.
            The pose vector is unknown and what we are looking to optimize.
            """
            cam_x = pose[0]
            cam_y = pose[1]
            cam_z = pose[2]
            roll = pose[3]
            pitch = pose[4]
            yaw = pose[5]

            r_axis = np.array([[1, 0, 0], 
                               [0, 0,-1], 
                               [0, 1, 0]])
            r_roll = np.array([[np.cos(roll), 0, -1*np.sin(roll)], 
                               [0, 1, 0], 
                               [np.sin(roll), 0, np.cos(roll)]])            
            r_pitch = np.array([[1, 0, 0], 
                                [0, np.cos(pitch), np.sin(pitch)], 
                                [0, -1*np.sin(pitch), np.cos(pitch)]])           
            r_yaw = np.array([[np.cos(yaw), -1*np.sin(yaw), 0, 0], 
                              [np.sin(yaw), np.cos(yaw), 0, 0], 
                              [0, 0, 1, 0]])
            T = np.array([[1, 0, 0, -cam_x], 
                          [0, 1, 0, -cam_y], 
                          [0, 0, 1, -cam_z], 
                          [0, 0, 0, 1]])
            C = r_axis @ r_roll @ r_pitch @ r_yaw @ T   
            
            
            if pts.ndim <= 1:
                pts = pts[np.newaxis,:]
            pts = (np.c_[pts, np.ones(pts.shape[0])]).T
            
            return C @ pts
               
    def projective_transform(self,rot_pt):
        """  
        This function performs the projective transform on generalized coordinates in the camera reference frame.
        This function needs the outputs of the rotational transform function (the rotated points).
        """
        focal = self.f 
        sensor = self.c  
        rot_pt = rot_pt.T
        #General Coordinates
        gcx = rot_pt[:,0]/rot_pt[:,2]
        gcy = rot_pt[:,1]/rot_pt[:,2]
        #Pixel Locations
        pu = gcx*focal + sensor[0]/2.
        pv = gcy*focal + sensor[1]/2.
        return np.array([pu,pv]).T
          
 
    def estimate_pose(self):
        
         def residual_pose(pose, realgcp, imagegcp,self):
            pt = self.projective_transform(self.rotational_transform(realgcp, pose))
            res = pt.flatten() - imagegcp.flatten()
            return res 

         for i in range(len(self.images)):
            realgcp = self.images[i].realgcp
            imagegcp = self.images[i].imagegcp
            self.images[i].pose = so.least_squares(residual_pose, self.images[i].pose, method='lm',args=[realgcp, imagegcp,self]).x
     
    def estimate_RWC(self):

        if(len(self.images) < 2):
            print("There are not 2 images in this camera class")
        #===========================
        def residual_RWC( RWC, pose1, pose2,imcor1,imcor2,self):
        
            pt_1 = self.projective_transform(self.rotational_transform(RWC, pose1)) # u,v based on first image
            pt_2 = self.projective_transform(self.rotational_transform(RWC, pose2)) 
            res_1 = pt_1.flatten() - imcor1.flatten()
            res_2 = pt_2.flatten() - imcor2.flatten()
            return np.hstack((res_1, res_2))  
        #==========================
    
        for i in range(len(self.images)):
            for j in range(len(self.images)):
                if( i != j):
                    self.images[i].realgcp = so.least_squares(residual_RWC, self.images[i].realgcp , method='lm',args=(self.images[i].pose,                         self.images[j].pose, self.images[i].imagegcp, self.images[j].imagegcp,self)).x
                    
	def pointCloud():
        	
            def triangulate(P0,P1,x1,x2):
                A = np.array([[P0[2,0]*x1[0] - P0[0,0], P0[2,1]*x1[0] - P0[0,1], P0[2,2]*x1[0] - P0[0,2], P0[2,3]*x1[0] - P0[0,3]],
                [P0[2,0]*x1[1] - P0[1,0], P0[2,1]*x1[1] - P0[1,1], P0[2,2]*x1[1] - P0[1,2], P0[2,3]*x1[1] - P0[1,3]],
                  [P1[2,0]*x2[0] - P1[0,0], P1[2,1]*x2[0] - P1[0,1], P1[2,2]*x2[0] - P1[0,2], P1[2,3]*x2[0] - P1[0,3]],
                  [P1[2,0]*x2[1] - P1[1,0], P1[2,1]*x2[1] - P1[1,1], P1[2,2]*x2[1] - P1[1,2], P1[2,3]*x2[1] - P1[1,3]]])
                    u,s,vt = np.linalg.svd(A)
                return vt[-1]
            
        I1 = self.images[0]
        I2 = self.images[1]
		h,w,d = I1.shape
		sift = cv2.xfeatures2d.SIFT_create()
		kp1,des1 = sift.detectAndCompute(I1,None)
		kp2,des2 = sift.detectAndCompute(I2,None)
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
    
		u1 = np.array(u1)  #general coords
		u2 = np.array(u2)

		#Make homogeneous
		u1 = np.c_[u1,np.ones(u1.shape[0])]
		u2 = np.c_[u2,np.ones(u2.shape[0])cu = w//2
		
		cv = image[0].h/2
        cu = image[0].w/2

		K_cam = np.array([[f,0,cu],[0,f,cv],[0,0,1]])
		K_inv = np.linalg.inv(K_cam)
		x1 = u1 @ K_inv.T  #camera coords
		x2 = u2 @ K_inv.T 

		E, inliers = cv2.findEssentialMat(x1[:,:2],x2[:,:2],np.eye(3),method=cv2.RANSAC,threshold=1e-3)
		inliers = inliers.ravel().astype(bool) 
		n_in,R,t,_ = cv2.recoverPose(E,x1[inliers,:2],x2[inliers,:2])

		P0 = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])
		P1 = np.hstack((R,t))
        
        for i in range(len(u2)):
            self.pointcloud.append(triangulate(P0,P1,x1[i],x2[i]))   #appends to list of points in xyz coordinates
                   
		self.pointCloud = np.array(self.pointCloud)
        self.pointCloud /= self.pointCloud[0][3]
                   
        def plotPointCloud():
            %matplotlib notebook
            fig = plt.figure()
            ax = fig.add_subplot(111,projection='3D')
            #for i in range(len(self.pointCloud)):
            ax.plot(*self.pointCloud.T,'k.')
	
                   
                   
                   
if __name__ == "__main__":
  Image1 = Image(sys.argv[1])
  Image2 = Image(sys.argv[2])
  pointCloud = camClass()
  pointCloud.add_images(Image1)
  pointCloud.add_images(Image2)
  pointCloud.pointCloud()
  pointCloud.plotPointCloud()