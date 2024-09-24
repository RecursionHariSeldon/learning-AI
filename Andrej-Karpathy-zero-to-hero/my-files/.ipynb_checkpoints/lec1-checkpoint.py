import math
import numpy as np
import matplotlib.pyplot as plt

from graphviz import Digraph

import torch

import random

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
        other = other if isinstance(other,Value) else Value(other)
        out =  Value(self.data + other.data ,   (self,other)  ,  '+')
        def _backward():
            self.grad += out.grad * 1.0
            other.grad += out.grad * 1.0
        out._backward = _backward
        return out

    def __radd__(self,other):
        return self + other
        
    def __mul__(self,other):
        other = other if isinstance(other,Value) else Value(other)
        out = Value(self.data * other.data ,   (self,other)  ,  '*')
        def _backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data
        out._backward = _backward
        return out

    
    def __rmul__( self,other):
        return self*other

    def __pow__(self,other):
        assert isinstance(other,(int,float) ), "Only supports int and float powers for now"
        out = Value( self.data**other , (self,) , f"**{other}" ) # here it's important to note that
                                                                 # despite having other , we do not pass it
                                                                 # into the childeren because it is of type int/float
                                                                 # hence will not be a part of back propagation
                                                                 # since it lacks properties of Value
  
        def _backward():
            self.grad += out.grad *  other * (self.data**(other -1 ) ) 
        out._backward = _backward
        return out
        

    def __truediv__(self,other):
        return self * (other ** -1)

    def __neg__(self):
        return self * -1

    def __sub__(self,other):
        return self + (-other)

    def exp(self):
        out = Value( math.exp(self.data) , (self,) , 'e^x')
        def _backward():
            self.grad += out.grad * out.data
        out._backward = _backward
        return out
        
    def tanh(self):
        x = self.data
        t =  ( math.exp(2*x) -1 )/(math.exp(2*x) +1) 
        out = Value( t  , (self,) , 'tanh')
        def _backward():
            self.grad += (1.0 - t**2) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()
        
        def topo_sort(n): # node n
            if n not in visited:
                visited.add(n)
                for child in n._prev:
                    topo_sort(child)
                topo.append(n)
        # we will call this on the output so first need to set its gradient to 1.0
        self.grad = 1.0
        topo_sort(self)
        for i in reversed(topo):
            i._backward()

class Neuron:
    def __init__(self,nin):
        self.w = [Value(random.uniform(-1,1)) for i in range(nin) ]
        self.b =  Value(random.uniform(-1,1)) 

    def __call__(self,x):
        wi_xi = zip( self.w , x )  # basically creates a list of tuples with the i-th yuple containing the ith w and x 
        act = sum((wi*xi for wi,xi in wi_xi) , self.b )
        out = act.tanh()
        return out

    def params(self):
        return self.w + [self.b]


class Layer():
    def __init__(self, nin,nout):
        self.neurons = [ Neuron(nin) for i in range(nout) ] 

    def __call__ (self, xin): 
        outs = [ n(xin) for n in self.neurons]
        return outs[0] if len(outs)==1 else outs

    def params(self):
        return [ p for neuron in self.neurons for p in neuron.params() ]


class MLP:
    def __init__(self, nin , nouts):
        all = [nin] + nouts 
        self.layers = [ Layer(all[i] , all[i+1] ) for i in range(len(nouts)) ]

    def __call__(self, xin):
        for layer in self.layers:
            xin = layer(xin)  # Smart ass way to update x , and call each of the layers along with it HAHA! KARPATHY FTW
        return xin

    def params(self):
        return [ p for layer in self.layers for p in layer.params() ]

# we move n here so the params don't change each time we run the code


