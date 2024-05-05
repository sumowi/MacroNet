from monet.example import parabolaA_B_C,func_pla,ddf_w1_w2_b_pla,funcspace_json
from monet.defdef import DefDefObj
from monet.flowfunc import FuncModel as FlowFunc
# from monet import Layer as Mix
from typing import OrderedDict
class MoNetInitial:
    def __init__(self,funcspace={}) -> None:
        self.funcspace = OrderedDict(funcspace)
        self.defdef = DefDefObj(self.funcspace,info_display=True)

    def ddf(self,func,call=None):
        """defdef a function,which return a callable function
        >>> from monet.base import MoNetInitial
        >>> m = MoNetInitial()
        >>> from monet.example import pla2_Type,func_pla
        >>> m.ddf(pla2_Type)
        "add 'pla2_Type' to funcspace"
        >>> m.ddf("pla__w_b_p1",lambda _w,b,p: lambda _x: func_pla(_w,_x,b,p))
        "add 'pla__w_b_p1' to funcspace"
        >>> m.ddf("pla__w_b_p2",lambda _x,_w,b,p: func_pla(_w,_x,b,p))
        "add 'pla__w_b_p2' to funcspace"
        >>> @m.ddf
        ... def logN(x, N=10):
        ...     "Define a logarithm with a base of N"
        ...     assert N > 0 and x > 0
        ...     print(f"def log{N}::logN")
        ...     return log(x, N)
        """
        return self.defdef(func,call)

    def f():
        """get a function from defdef functionspace.
        """
        return ...

    def layer():
        return ...
    def cell():
        return ...
    def Net():
        return ...
# class DefDef(DefDefObj):
#     """make a normal func to ddf in funcspace or add a ddf func to funcspace
#     funcspae store the func information. if give a func to add, it must add to funcspace firstly。
#     then, while called a alias func base on funcspace, a func will be created in namespace.
#     """
#     def __init__(self, funcspace = {}) -> None:
#         super().__init__(funcspace)
#         """The basic usage:
#         >>> from monet import DefDef
#         >>> mn = DefDef() # first create a ddf object
#         """
#         # init a DefDef object, can load a funcspace
#         # if not give funcspace, will load a blank space
#         self.funcspace = funcspace
#         self.tempspace = {}


#         self.pla = self.func_pla = func_pla
#         self.pla_w1_w2_b= self.ddf_w1_w2_b_pla =ddf_w1_w2_b_pla
#         self.parabolaA_B_C = parabolaA_B_C
#         self.funcspace_json = funcspace_json

# class MoNet:
#     def __init__(self, funcspace = {}) -> None:
#         pass

# def MoNetInitial(funcspace = {}):
#     funcspace = funcspace
#     # tempspace 则是独立的
#     ddf = DefDef(funcspace)
#     fn = FlowFunc(funcspace)
#     mn = MoNet(funcspace)
#     return ddf,fn,mn

if __name__ == "__main__":
    import doctest
    doctest.testmod()