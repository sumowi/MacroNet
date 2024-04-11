"""
>>> from monet.example import funcspace_json
>>> ddf_pla = funcspace_json['ddf_w1_w2_b_pla']['func']
>>> nand = ddf_pla(-0.5,-0.5,0.7)
>>> nand([0,1])>0,nand([1,1])>0,nand([0,0])>0,nand([1,0])>0
(True, False, True, True)
>>> ddf_pba = funcspace_json['parabolaA_B_C']['func']
>>> parabola = ddf_pba(1,2,3)
>>> parabola(0),parabola(1),parabola(2),parabola(3)
(3, 6, 11, 18)
"""


funcspace_dict_full = {
    "ddf_w1_w2_b_pla": {
        "default": {
            "w1": 0,
            "w2": 1,
            "b": 0},
        "help": "This is a function that returns the perceptron function.",
        "func": lambda w1,w2,b: lambda _x:
            func_pla(_w=[w1,w2],_x=_x,b=b,alpha=0)
    },
    "parabolaA_B_C":{
        "default": {
            "A": 1,
            "B": 2,
            "C": 3
        },
        "help": "Define a parabola fun: y=A*x*x+B*x+C",
        "func": lambda A,B,C: lambda x:
            parabola(x, A=A, B=B, C=C)
    }
}

funcspace_dict_name={
    "ddf_w1_w2_b_pla": lambda w1=0,w2=0,b=0: lambda _x:
        func_pla(_w=[w1,w2],_x=_x,b=b,alpha=0),
    "parabolaA_B_C": lambda A=1,B=2,C=3: lambda x:
        parabola(x, A=A, B=B, C=C)
}

funcspace_dict_value={
    "pla_0_0_0": lambda w1,w2,b: lambda _x:
        func_pla(_w=[w1,w2],_x=_x,b=b,alpha=0),
    "parabola1_2_3": lambda A,B,C: lambda x:
        parabola(x, A=A, B=B, C=C)
}

def func_pla(_w=[0,1],_x=[0,1],b=0,alpha=0.01):
    """This is a perceptron function.
    y = prelu(sum(_w*_x)+b)
    where _w, _x is a vector, y, b is a scalar.
    prelu is the activation function with a alpha default to 0.01.
    >>> from monet.example import func_pla
    >>> func_pla(_w=[0.5,0.5],_x=[0,1],b=-0.7,alpha=0)>0 # [AND]
    False
    >>> func_pla(_w=[0.5,0.5],_x=[0,1],b=-0.2,alpha=0)>0 # [OR]
    True
    >>> func_pla(_w=[-0.5,-0.5],_x=[0,1],b=0.7,alpha=0)>0 # [NAND]
    True
    """

    def prelu(x, alpha=0.01):
        return x if x>0 else alpha*x

    def func(_x):
        y =[w*x for w,x in zip(_w,_x)]
        y = sum(y)+b
        y = prelu(y, alpha)
        return y

    return func(_x) # normal func return func called with _x

def pla_Type(_x=[0,1],Type='NAND'):
    """This is a perceptron function.
    y = prelu(sum(_w*_x)+b)
    where _w, _x is a vector, y, b is a scalar.
    prelu is the activation function with a alpha default to 0.01.
    >>> from monet.example import pla2_Type
    >>> pla2_Type([0,1],'AND'),pla2_Type([0,1],'OR'),pla2_Type([0,1],'NAND')
    (False, True, True)
    """
    wb={
        "AND":[[0.5,0.5],-0.7],
        "OR":[[0.5,0.5],-0.2],
        "NAND":[[-0.5,-0.5],0.7]
    }
    _w,b=wb[Type]

    return func_pla(_w,_x,b)>0


def pla2_Type(_x=[0,1],Type='NAND'):
    return pla_Type(_x,Type)

def ddf_w1_w2_b_pla(w1=0,w2=1,b=0):
    """This is a function that returns the perceptron function.
    >>> from monet.example import ddf_w1_w2_b_pla
    >>> nand = ddf_w1_w2_b_pla(w1=-0.5,w2=-0.5,b=0.7)
    >>> nand([0,1])>0,nand([1,1])>0,nand([0,0])>0,nand([1,0])>0
    (True, False, True, True)
    """
    def func(_x=[0,1]):
        y = func_pla(_w=[w1,w2],_x=_x,b=b,alpha=0)
        return y
    return func # ddf fun return func

def parabola(x, A, B, C):
    """Define a parabola fun: y=A*x*x+B*x+C
    >>> from monet.example import parabola
    >>> parabola(1, A=1, B=2, C=3)
    6
    """
    return A * x * x + B * x + C

def parabolaA_B_C(x, A=1, B=2, C=3):
    """Define a parabola: y=A*x*x+B*x+C.
    Where you want the A,B,C can be bound at func name.
    >>> from monet.example import parabolaA_B_C
    >>> parabolaA_B_C(1)
    6
    """
    return A * x * x + B * x + C


if __name__ == "__main__":
    import doctest
    doctest.testmod()