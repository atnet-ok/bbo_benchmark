import numpy as np
import sys

from abc import ABCMeta,abstractmethod
from typing import List, Dict,Tuple


# https://qiita.com/nabenabe0928/items/08ed6495853c3dd08f1e
class BecnmarkFunction(metaclass=ABCMeta):
    @abstractmethod
    def evaluation(self,params:List[any])->float:
        pass

    @abstractmethod
    def get_params_range(self)->List[Tuple[float]]:
        pass

class QuadraticFunction(BecnmarkFunction):
    def __init__(self,dim=10,scale = 10):
        self._dim = dim
        self._range = (-1*scale,1*scale)

    def evaluation(self,params):
        if len(params) != self._dim:
            print('Error: \'params\' must be {} dimentions'.format(self._dim), file=sys.stderr)
            sys.exit(1)
        else:
            params = np.array(params)
            eval = np.sum(params*params)
            return eval

    def get_params_range(self):
        range_s = [self._range for i in range(self._dim)]
        return range_s

class AckleyFunction(BecnmarkFunction):
    def __init__(self,dim=10,scale = 30):
        self._dim = dim
        self._range = (-1*scale,1*scale)
        self.boundaries = np.array(self._range)

    def evaluation(self, params):
        x = np.array(params)
        t1 = 20
        t2 = - 20 * np.exp(- 0.2 * np.sqrt(1.0 / len(x) * np.sum(x ** 2)))
        t3 = np.e
        t4 = - np.exp(1.0 / len(x) * np.sum(np.cos(2 * np.pi * x)))
        return t1 + t2 + t3 + t4

    def get_params_range(self):
        range_s = [self._range for i in range(self._dim)]
        return range_s
class RosenbrockFunction(BecnmarkFunction):
    def __init__(self,dim=10,scale = 5):
        self._dim = dim
        self._range = (-1*scale,1*scale)
        self.boundaries = np.array(self._range)

    def evaluation(self, params):
        x = np.array(params)
        val = 0
        for i in range(0, len(x) - 1):
            t1 = 100 * (x[i + 1] - x[i] ** 2) ** 2
            t2 = (x[i] - 1) ** 2
            val += t1 + t2
        return val
        
    def get_params_range(self):
        range_s = [self._range for i in range(self._dim)]
        return range_s
class StyblinskiTangFunction(BecnmarkFunction):
    def __init__(self,dim=10,scale = 5):
        self._dim = dim
        self._range = (-1*scale,1*scale)
        self.boundaries = np.array(self._range)

    def evaluation(self, params):
        x = np.array(params)
        t1 = np.sum(x ** 4)
        t2 = - 16 * np.sum(x ** 2)
        t3 = 5 * np.sum(x)
        return 0.5 * (t1 + t2 + t3) + 39.166165*self._dim

    def get_params_range(self):
        range_s = [self._range for i in range(self._dim)]
        return range_s
class GriewankFunction(BecnmarkFunction):
    def __init__(self,dim=10,scale = 5):
        self._dim = dim
        self._range = (-1*scale,1*scale)
        self.boundaries = np.array(self._range)

    def evaluation(self, params):
        x = np.array(params)
        w = np.array([1.0 / np.sqrt(i + 1) for i in range(len(x))])
        t1 = 1
        t2 = 1.0 / 4000.0 * np.sum(x ** 2)
        t3 = - np.prod(np.cos(x * w))
        return t1 + t2 + t3

    def get_params_range(self):
        range_s = [self._range for i in range(self._dim)]
        return range_s
class SchwefelFunction(BecnmarkFunction):
    def __init__(self,dim=10,scale = 500):
        self._dim = dim
        self._range = (-1*scale,1*scale)
        self.boundaries = np.array(self._range)

    def evaluation(self, params):
        x = np.array(params)
        return - np.sum(x * np.sin( np.sqrt( np.abs(x) ) ) ) + 418.9829*self._dim

    def get_params_range(self):
        range_s = [self._range for i in range(self._dim)]
        return range_s
class XinSheYangFunction(BecnmarkFunction):
    def __init__(self,dim=10,scale = 6):
        self._dim = dim
        self._range = (-1*scale,1*scale)
        self.boundaries = np.array(self._range)

    def evaluation(self, params):
        x = np.array(params)
        t1 = np.sum( np.abs(x) )
        e1 = - np.sum( np.sin(x ** 2) )
        t2 = np.exp(e1)
        return t1 * t2

    def get_params_range(self):
        range_s = [self._range for i in range(self._dim)]
        return range_s