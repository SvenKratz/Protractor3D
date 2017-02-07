'''
Created on Apr 12, 2011

@author: svenkratz
'''


import TestGestures
import Protractor3D.Protractor3D
from Protractor3D.Protractor3D import *


def triplify(g):
	out = []
	if len(g) % 3 != 0:
		print "Warning: Data not divisible by 3"
	for k in xrange(len(g)/3):
		out = out + [[g[3*k], g[3*k+1], g[3*k+2]]]
	return out


triangle = triplify(TestGestures.Triangle)

print triangle

circle1 = triplify(TestGestures.Circle_1)
circle2 = triplify(TestGestures.Circle_2)
rectangle = triplify(TestGestures.Rectangle)

p_triangle = Protractor3D(triangle)
p_circle1 = Protractor3D(circle1)
p_circle2 = Protractor3D(circle2)
p_rectangle = Protractor3D(rectangle)


#print p_circle1.trace
#
#print "Trace", p_triangle.trace
#print "Resampled", p_triangle.resampled
#print "Scaled", p_triangle.scaled
#print "Centered", p_triangle.centered
#print "Template", p_triangle.template


print "========== Evaluations =============="

Protractor3D.DEBUG = 5


gesturesAndNames = [(p_triangle,"Triangle"), (p_circle1,"Circle1"), ( p_circle2, "Circle2") , (p_rectangle, "Rectangle")]

while gesturesAndNames != []:
	gesture = gesturesAndNames[0]
	templates = gesturesAndNames[1:]
	gesturesAndNames = templates
	if len(templates) != 0:
		for t in templates:
			print "======================================="
			print "Results for", gesture[1]," <---> ", t[1]
			gesture[0].protractor3D_classify(gesture[0].template, t[0].template)


