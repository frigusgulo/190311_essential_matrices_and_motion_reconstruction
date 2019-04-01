#unage class
import camClass
import imageClass
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image
import matplotlib.image as mpimg
import scipy.optimize as so

class photogramClass():

	def __init__(self,camera1,camera2):
		self.camera1 = camera1
		self.camera2 = camera2

	def residual_RWC(self, self.camera1,self.camera2):
		