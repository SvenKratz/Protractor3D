'''
Created on Aug 6, 2010, Modifications starting April 2011

Implementation of Protractor3D Algorithm. 

Generic Implementation for GPL Use

@author: Sven Kratz

Mobile Interaction Group
University of Munich
Amalienstr. 17 
80333 Munich Germany
'''


import scipy, numpy, scipy.linalg

from sys import float_info
import traceback

from Protractor3DHelper import *
import sys




DEBUG = 3
SEP = "======="

#def test_protractor3D(dict):
#	'''tests the protractor with a gesture data dict'''
#	
#	subject = dict.values()[0]
#	gesture_classes = subject['gestures']
#	gesture_class = gesture_classes.items()[6]
#	print "Gesture Indices"
#	for c in gesture_classes.items():
#		print c[0],
#	print ""
#	print "Gid_a", gesture_class[0]
#	gestures_a = gesture_class[1]
#	
#	gesture_class_b = gesture_classes.items()[7]
#	print "Gid_b", gesture_class_b[0]
#	gestures_b = gesture_class_b[1]
#	g = gestures_a[3]
#	t = gestures_b[5]
#	PATH_SIZE = 32
#	CUBE_SIZE = 100
#	p1 = Protractor3D(g, PATH_SIZE, CUBE_SIZE)
#	p2 = Protractor3D(t, PATH_SIZE, CUBE_SIZE)
#	print SEP
#	print "Starting Classification"
#	p1.protractor3D_classify(p1.template, p2.template)
#	#print SEP
#	#print "MATLAB POINTS"
#	#print_matlab_array(p1.template)
#	#print SEP
#	# print_matlab_array(p2.template)
	
def resample(trace, segment_amount = 32):
	''' resamples trace to have N segments. adapted from $1 Recognizer (Wobbrock) '''
	trace = list(trace)
	resampled = [trace[0]]
	N = segment_amount
	p_length = path_length(trace)
	if DEBUG>1:print "Path_Length", p_length, "Trace Length", len(trace)
	increment = p_length / N
	if DEBUG>1:print "Increment", increment
	D = 0.0
	i = 0
	trace_length = len(trace)
	#length_limit = trace_length - 1
	#length_limit = trace_len
	
	while i < trace_length - 1:
		i = i + 1
		
		pp = trace[i-1]
		p = trace[i]
		
		diff = subtract3(p,pp)
		
		# distance between current and previous point
		d = norm3( diff )
		
		# conditions of the algorithm
		if D+d > increment:
			over = abs(D-increment)
			#if DEBUG: print "\t Over", over, "trc_length", trace_length,"d", "IDX", i, 
			# direction vector of overstep + scalingv and get to new point (cf. Wobbrock)
			q = [	pp[0] +( diff[0] / d ) * over, 
					pp[1] +( diff[1] / d ) * over, 
					pp[2] +( diff[2] / d ) * over]
			
			resampled.append(q)
			
			# Reset D
			# q is next point
			trace.insert(i, q)
			trace_length = trace_length + 1
			D = 0.0
			
			
			
		elif D+d == increment:
			# just add current point and increment
			resampled.append(p)
			# reset D
			D = 0.0
		else:
			D  = D + d
	r_length = len(resampled)
	if r_length-segment_amount > 0:
		diff = r_length-segment_amount
		resampled = resampled[:-diff]		
	return numpy.array(resampled)
			

def center(points):
	''' calculates centroid of points and subtracts centroid from points '''
	ctroid = centroid(points)
	return [subtract3(v, ctroid) for v in points]

def centroid(points):
	''' calculates centroid of points '''
	# reduce adds up the elements, x is element from points, y is accumulator
	sum = reduce(lambda x,y: [x[0] + y[0], x[1]+ y[1], x[2]+ y[2]], points)
	if DEBUG>1: print "Sum...", sum
	l = len(points)
	# return the centroid (normalize with l)
	return [sum[0]  / l, sum[1] /l, sum[2] / l]
	

	
def make_gesture_trace(dataPoints):
	''' generates a gesture trace (raw datapoints from shake device)'''
	''' expects a list of triplets, i.e. x,y,z accelerations ''' 
	v = dataPoints
	length = len(dataPoints)
	tp = numpy.zeros(3*length).reshape(length, 3)
	
	#print dataPoints.shape, tp.shape
	
	tp[0] = dataPoints[0]
	
	for i in xrange(1,len(dataPoints)):
		tp[i] = tp[i-1] + dataPoints[i]
	
	print tp
	return tp

def path_length(points):
	'''calculates length of trace (path)'''
	length = 0.0
	for i in xrange(1, len(points)):
		p = points[i]
		pp = points[i-1]
		length = length + norm3([p[0]-pp[0], p[1]-pp[1], p[2]-pp[2]] )
	return length

def scale_to_cube(points, CUBE_SIZE = 100):
	'''scales points to fit into a cube of predefined size'''
	s = CUBE_SIZE

	# define min and max flotas
	min = [0+float_info[0],0+float_info[0],0+float_info[0]] 
	max = [0+float_info[3],0+float_info[3],0+float_info[3]] 
	
	for p in points:
		i = 0
		for c in p:
			if c < min[i]: 
				min[i] = c
			elif c >max[i]:
				max[i] = c
			i = i + 1
	if DEBUG>1: print "min / max", min, max
	
	# scale
	
	sc = [max[0] - min[0], max[1] - min[1], max[2] - min[2] ]
	if DEBUG>0: print "scaler", sc
	
	
	
	s1, s2 , s3 = (0.,0.,0.)
	
	# avoid division by 0! 
	if abs(sc[0]) >= 0.00000001:
		s1 = s / sc[0]
	else:
		s1 = 0.0
		
	if abs(sc[1]) >= 0.00000001:
		s2 = s / sc[1]
	else:
		s2 = 0.0
		
	if abs(sc[2]) >= 0.00000001:
		s3 = s / sc[2]
	else:
		s3 = 0.0

	print "=== s3 ",s3
	# another classic one-liner (performs the scaling)
	scaled_points = [[v[0] * (s1) , v[1] * (s2), v[2] * (s3)] for v in points ]
	
	return scaled_points

class Protractor3D:
	
	# amount of segments for templates after subsampling
	def __init__(self, raw_data = None, PATH_SIZE  = 32, CUBE_SIZE  = 100):

		self.PATH_SIZE = PATH_SIZE
		self.CUBE_SIZE = CUBE_SIZE
		self.raw = None
		self.trace = None
		self.resampled = None
		self.scaled = None
		self.centered = None
		self.template = None
		
		
		if raw_data != None:
			if DEBUG:
				print "[Protractor3D]: (initializing with raw data)"
			if DEBUG:
				if raw_data!= None: print "Raw Data Len",len(raw_data),
				print "PATH_SIZE",PATH_SIZE,"CUBE_SIZE",CUBE_SIZE
				print "[Protractor3D] Generating Template..."
			self.raw = raw_data
			self.generate_template(raw_data)
		
		
	
	def generate_template(self, dataPoints):
		''' generates a template for P3D from raw acc. data'''

		print dataPoints
		self.trace = make_gesture_trace(dataPoints)
		self.resampled = resample(self.trace, self.PATH_SIZE) 
		self.scaled = scale_to_cube(self.resampled, self.CUBE_SIZE)
		self.centered = center(self.scaled)
		self.template = self.centered
		if DEBUG>1: 
			print SEP
			print "Template generated..."
			print "Template Points"
			print_points(self.template)
			print "Template Length", len(self.template)
			print SEP
	
	def protractor3D_classify(self, gesture, template):
		''' assumes resampled, scaled and centered templates '''
		g = gesture
		t = template
		Sxx= 0.0
		Sxy= 0.0
		Sxz= 0.0 
		Syx= 0.0 
		Syy= 0.0 
		Syz= 0.0 
		Szx= 0.0 
		Szy= 0.0 
		Szz= 0.0
	
		# contstruct Matrix M
		# add up the scalar product to entries of M
		
		
		
		t_length = len(t)
		g_length = len(g)
		
		print "l_gesture, l_template", g_length, t_length
		
		assert len(t) > 0 and len(g) > 0, "Length(s) zero!"
		
		length = 0
		
		if g_length >= t_length:
			length = t_length
		else:
			length = g_length
	
		for i in xrange(length):
			Sxx = Sxx + g[i][0] * t[i][0];
			Sxy = Sxy + g[i][0] * t[i][1];
			Sxz = Sxz + g[i][0] * t[i][2];
			Syx = Syx + g[i][1] * t[i][0];
			Syy = Syy + g[i][1] * t[i][1];
			Syz = Syz + g[i][1] * t[i][2];
			Szx = Szx + g[i][2] * t[i][0];
			Szy = Szy + g[i][2] * t[i][1];
			Szz = Szz + g[i][2] * t[i][2];
		
		# compute N
		# Horn 1978, p.635
		
		N_ = [[0.0]*4]*4
		# initialize numpy array from N_ and set to double datatype
		N = numpy.array(N_)
		
#===============================================================================
#		Original MatLab Source:
#		N(1,1) = Sxx + Syy + Szz
#		N(1,2) = Syz - Szy;
#		N(1,3) = Szx - Sxz;
#		N(1,4) = Sxy - Syx;
# 
#		N(2,1) = Syz - Szy;
#		N(2,2) = Sxx - Syy - Szz;
#		N(2,3) = Sxy + Syx;
#		N(2,4) = Szx + Sxz;
# 
#		N(3,1) = Szx - Sxz;
#		N(3,2) = Sxy + Syx;
#		N(3,3) = -Sxx + Syy - Szz;
#		N(3,4) = Syz + Szy;
# 
#		N(4,1) = Sxy - Syx;
#		N(4,2) = Szx + Sxz;
#		N(4,3) = Syz + Szy;
#		N(4,4) = -Sxx - Syy + Szz;
#===============================================================================

		N[0][0] = Sxx + Syy + Szz
		N[0][1] = Syz - Szy;
		N[0][2] = Szx - Sxz;
		N[0][3] = Sxy - Syx;

		N[1][0] = Syz - Szy;
		N[1][1] = Sxx - Syy - Szz;
		N[1][2] = Sxy + Syx;
		N[1][3] = Szx + Sxz;

		N[2][0] = Szx - Sxz;
		N[2][1] = Sxy + Syx;
		N[2][2] = -Sxx + Syy - Szz;
		N[2][3] = Syz + Szy;

		N[3][0] = Sxy - Syx;
		N[3][1] = Szx + Sxz;
		N[3][2] = Syz + Szy;
		N[3][3] = -Sxx - Syy + Szz;
		
		if DEBUG>1: print "N-Matrix", N
		
		
		# d = eigenvects
		# v = eigenvals
		
		# test for matrix singularity
		det = scipy.linalg.det(N)
		if det == 0:
			print "#### Matrix is singular, stopping! ####"
			return 0.0, sys.float_info.max, sys.float_info.max,0,0
		
		
		
		[V,D] = scipy.linalg.eigh(N)
		
		# identify the largest eigenvalue and pick the corresponding eigenvector
		if DEBUG >1:
			print "Eigenvects", D
			print "Eigenvals", V
		largest_eigenval = V.argmax()
		# needs to be transposed to get the same eigenvector as in MatLab
		q = D.transpose()[largest_eigenval]
		
		if DEBUG>1: print "Largest Eigenval q:", largest_eigenval, "largest eigenvect q:", q
		
		# q is a quaternion of the form q = [qw, qx, qy,qz]
		# q = cos(theta/2) + sin(theta/2) * (i qx + j qy + k qz)
		
		# compute the rotation angle from q
		
		theta = 2.0*numpy.arccos(q[0])
		if DEBUG>1: print "THETA",theta,"=",theta*180/ math.pi ,"deg" 
		
		# compute axis of rotation by normalizing (qx, qy, qz)
		
		
		axis = q[1:]
		axis = axis / numpy.linalg.norm(axis) 
		
		if DEBUG>1: print "Axis", axis, "norm", numpy.linalg.norm(axis)
		
		# rotate points back
		R = rotation_matrix3(axis, theta)
		if DEBUG >1:
			print "Rotation Matrix"
			print R
		# numpy.dot is the standard matrix multiplication
		back_rotated_points = [numpy.dot(R,v) for v in t]
		
		
		# mse between gesture and template
		
		mse1 = mse(g, t) 
		print "Theta:", theta*180/ math.pi,
		print "Error:  gesture <> template:", mse1,
		
		
		mse2 = mse(g, back_rotated_points)
		print "rotated gesture <> template:", mse2
		if DEBUG > 1:
			try:
				mse3 = mse(t, back_rotated_points)
				print "template <> rotated points:", mse3,
			except:
				traceback.print_exc()
		#sp1 = scalar(g,t)
		#sp2 = scalar(t, back_rotated_points)
		#print "scalar g <> t",sp1, "scalar g <> r(t)", sp2
		return theta*180/ math.pi,mse2,mse1,0,0
		# TODO plot stuff!!!!!
		
		
		
		
		
		
		
		
		
		
		
		  
	
	
		
	
	
	
	
	