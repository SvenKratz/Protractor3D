'''
Created on Aug 6, 2010

@author: Sven Kratz
(c) 2010, 2011, Sven Kratz, University of Munich

# helper module with maths functions



'''
import math
import numpy

SEP = "============================================"

def print_matlab_array(points):
	print SEP
	print "P=["
	for p in points:
		for pp in p:
			print pp,",",
		print ";"
	print "];"

def print_points(points):
	for i in xrange(len(points)):
		print i, points[i]

def norm3(v):
	'''euclidean norm of vector v, v in R^3'''
	return math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])


def subtract3(v1,v2):
	'''subtract v2 from v1 (v1,v2 in R^3)'''
	return [ v1[0] - v2[0], v1[1] - v2[1], v1[2] - v2[2] ]

def rotate_points(axis, angle, points):
	l = norm3(axis)
	axis[0]/=l
	axis[1]/=l
	axis[2]/=l
	R = rotation_matrix3(axis,angle)
	rotated_points  = [numpy.dot(R,v) for v in points]
	return rotated_points

def rotation_matrix3(axis, angle):
	
	''' returns a rotation matrix from axis and angle'''
	
	s = numpy.sin(-angle)
	c = numpy.cos(-angle)
	x = axis[0]
	y = axis[1]
	z = axis[2]
	
	# rotation matrix from angle and axis
	#M = [(1+(1-c))*(x*x-1) -z*s+(1-c)*x*y y*s+(1-c)*x*z; 
	#   z*s+(1-c)*x*y 1+(1-c)*(y*y*-1) -x *s+(1-c)*y*z;
	#   -y*s+(1-c)*x*z x*s+(1-c)*y*z 1+(1-c)*(z*z-1)];
	M = numpy.array([[x*x+(1-x*x)*c, x*y*(1-c)-z*s, x*z*(1-c)+y*s],
		[x*y*(1-c)+z*s, y*y+(1-y*y)*c, y*z*(1-c)-x*s],
		[x*z*(1-c)-y*s, y*z*(1-c)+x*s, z*z+(1-z*z)*c]], dtype=numpy.dtype('d'))
	return M

def scalar(p, q):
	sp  = 0
	if len(q) >= len(p):
		n = len(p)
	else:
		n = len(q)
	for i in xrange(n):
		sp+=p[i][0]*q[i][0]+p[i][1]*q[i][1]+p[i][2]*q[i][2]
	return sp

def mse(points1, points2):
	error = 0.0

	if len(points2) >= len(points1):
		n = len(points1)
	else:
		n = len(points2)
	for i in xrange(n):
		d = subtract3(points1[i],points2[i])
		error = error + (d[0]*d[0] + d[1]*d[1] + d[2]*d[2])
	return error / n


	
		
		
  