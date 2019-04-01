from camClass_mint import *



I_1 = plt.imread('campus_stereo_1.jpg')
I_2 = plt.imread('campus_stereo_2.jpg')

gcp_1 = np.loadtxt('gcp_stereo_1.txt',delimiter=',')
gcp_2 = np.loadtxt('gcp_stereo_2.txt',delimiter=',')

#Estiamte the pose of the two cameras
Image1 = Image(I_1,gcp_1[:,:2],gcp_1[:,2:])
Image2 = Image(I_2,gcp_2[:,:2],gcp_2[:,2:])

# Initial guess at pose
roll_init = np.radians(0.0)
pitch_init = np.radians(0.0)
yaw_init = np.radians(90.0)
pose_0 = np.array([2.725e5, 5.1939e6, 1100., roll_init, pitch_init, yaw_init])

# Initialize camera object
myCam = camClass((27/36.)*3264,2448.,3264.,pose_0)
myCam.add_images(Image1)
myCam.add_images(Image2)
myCam.estimate_pose()

myCam.images[0].imagegcp = np.array([2013, 1288])
myCam.images[1].imagegcp = np.array([1201, 1216])

#Initial Guess at real world coordinates of clock tower face
X_ct_0 = np.array([2.725e5, 5.1939e6, 1100.])
myCam.images[0].realgcp = X_ct_0
myCam.images[1].realgcp = X_ct_0

# Estimate Position of Object
myCam.estimate_RWC()
print ('Optimal Coordinates of Clock Tower: 12 T %.2f %.2f'%(myCam.images[0].realgcp[0],myCam.images[0].realgcp[1]))
print ('Elevation: %.2f meters'%myCam.images[0].realgcp[2])
