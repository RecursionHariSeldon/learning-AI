import math
import numpy as np
import matplotlib.pyplot as plt 

# now this is one thing which won't be explained in full detail, just go with it , coz it doesn't jave anything to do 
# with the working of the neural net , just used to visualize stuff to make the creation process easier

# just node that it creates extra nodes to represent the operands used to create the node
# also understand how the formatting works to print text in one cell with label, data , grad
from graphviz import Digraph

def trace(root):
  # builds a set of all nodes and edges in a graph
  nodes, edges = set(), set()
  def build(v):
    if v not in nodes:
      nodes.add(v)
      for child in v._prev:
        edges.add((child, v))
        build(child)
  build(root)
  return nodes, edges

def draw_dot(root):
  dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'}) # LR = left to right
  
  nodes, edges = trace(root)
  for n in nodes:
    uid = str(id(n))
    # for any value in the graph, create a rectangular ('record') node for it
    dot.node(name = uid, label = "{ %s | data %.4f }" % (n.label, n.data), shape='record')
    if n._op:
      # if this value is a result of some operation, create an op node for it
      dot.node(name = uid + n._op, label = n._op)
      # and connect this node to it
      dot.edge(uid + n._op, uid)

  for n1, n2 in edges:
    # connect n1 to the op node of n2
    dot.edge(str(id(n1)), str(id(n2)) + n2._op)

  return dot


########################### STARTS HERE ###############################
class Value:
    def __init__(self , a , _children =() ,_op = '' ,l=''):
        self.data = a
        self._prev = set(_children)
        self._op = _op
        self.label = l
        self.grad = 0
        self._backward = lambda:None

    def __repr__(self):
        return f"Value = {self.data}"

    def __add__(self,other,):
        out =  Value(self.data + other.data ,   (self,other)  ,  '+')
        def _backward():
            self.grad = out.grad * 1.0
            other.grad = out.grad * 1.0
        out._backward = _backward
        return out
        
    def __mul__(self,other):
        out = Value(self.data * other.data ,   (self,other)  ,  '*')
        def _backward():
            self.grad = out.grad * other.data
            other.grad = out.grad * self.data
        out._backward = _backward
        return out
        
    def tanh(self):
        out = Value( np.tanh(self.data) , (self,) , 'tanh')
        def _backward():
            self.grad = (1.0 - out.data**2) * out.grad
        out._backward = _backward
        return out

x1 = Value(0.7 , l = 'x1')
w1 = Value(0.30, l = 'w1')
x2 = Value(-0.20, l = 'x2')
w2 = Value(0.40, l = 'w2')
x3 = Value(-0.20, l = 'x3')
w3 = Value(0.9, l = 'w3')

x1w1 = x1*w1; x1w1.label = 'x1w1'
x2w2 = x2*w2; x2w2.label = 'x2w2'
x3w3 = x3*w3; x3w3.label = 'x3w3'
d = x1w1*x3w3; d.label  = 'd'
b = d + x2w2;b.label = 'b'
c = b.tanh(); c.label ='c'

c.grad = 1.0
c._backward()

print(b.grad)