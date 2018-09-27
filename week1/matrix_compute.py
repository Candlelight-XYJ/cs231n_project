import numpy as np


class matrix(object):
    def __init__(self,name):
        self.name=name

    def matrix_multiply(self,x,y):
        result=np.dot(x,y)
        return result

#### example
x=np.array([[1,2],[3,4]])
print("x is",x)
y=np.array([[5,6],[7,8]])
print("y is ",y)
g=matrix('test')
print("x*y is ")
print(g.matrix_multiply(x,y))
