from typing import Any, Callable, overload
from monet.example import parabolaA_B_C,func_pla,ddf_w1_w2_b_pla,pla_Type,pla2_Type   # noqa: F401
from monet.example import funcspace_dict_full,funcspace_dict_name,funcspace_dict_value # noqa: F401

try:
    import torch
    from monet.torch_ddf import torch_dict # noqa: F401
except Exception:
    torch = None

class MoNetInitial:
    def __init__(self,funcspace={}) -> None:
        """f is an blank func, which return the input.
        >>> from monet import MoNetInitial
        >>> import torch.nn as nn
        >>> m = MoNetInitial(nn)
        >>> m.Linear(10,1).name
        '@ddf:Linear(in_features=10, out_features=1, bias=True)'
        >>> m.ReLU().name
        '@ddf:ReLU()'
        """
        from monet.defdef import DefDefObj
        from monet._monet import Layer
        from typing import OrderedDict

        self.funcspace = OrderedDict()
        self.namespace = OrderedDict()
        self.__funcspace__ = None
        self.defdef = DefDefObj(spaceobj=self)

        if isinstance(funcspace,dict):
            self.defdef.add(funcspace)
        else:
            self.__funcspace__ = funcspace

        self.f = self.defdef.get()
        self.find = self.defdef.find
        self.Layer = Layer
    @overload
    def ddf(self,func_name:str) -> Callable:...
    @overload
    def ddf(self,name:str,func:Callable) -> Callable:...
    @overload
    def ddf(self,dict:dict) -> Callable:...
    @overload
    def ddf(self,func:Callable) -> Callable:...
    def ddf(self,*args,**kwargs):
        """defdef a function,which return a callable function.
        >>> from monet import MoNetInitial
        >>> m = MoNetInitial()
        >>> from monet.example import pla2_Type,func_pla
        >>> m.ddf(pla2_Type).name
        '@ddf:pla2_Type'
        >>> m.ddf("pla__w_b_p1",lambda _w,b,p: lambda _x: func_pla(_w,_x,b,p)).name
        '@ddf:pla__w_b_p1'
        >>> m.ddf("pla__w_b_p2",lambda _x,_w,b,p: func_pla(_w,_x,b,p)).name
        '@ddf:pla__w_b_p2'
        >>> @m.ddf
        ... def logN(x, N=10):
        ...     "Define a logarithm with a base of N"
        ...     assert N > 0 and x > 0
        ...     print(f"def log{N}::logN")
        ...     return log(x, N)
        >>> logN.name
        '@ddf:logN'
        """
        return self.defdef.add(*args,**kwargs)

    def fit(self,func_name):
        """call a function from defdef functionspace.
        >>> from monet import MoNetInitial
        >>> m = MoNetInitial()
        >>> from monet.example import pla2_Type
        >>> m.ddf(pla2_Type).name
        '@ddf:pla2_Type'
        >>> AND=m.fit("pla2_AND")
        ('pla2_Type', {'Type': 'AND'})
        >>> OR=m.fit("pla2_OR")
        ('pla2_Type', {'Type': 'OR'})
        >>> NAND=m.fit("pla2_NAND")
        ('pla2_Type', {'Type': 'NAND'})
        >>> XOR =(AND+OR)*(NAND+OR)*NAND
        >>> XOR([1,1]),XOR([0,0]),XOR([0,1]),XOR([1,0])
        (True, True, False, False)
        >>> (m.f*((AND,NAND),(OR,NAND)))([1,1])
        [[True, False], [True, False]]
        >>> (m.f*(AND,NAND)&(OR,NAND))([1,1])
        [True, False, True, False]
        >>> (m.f*(AND,NAND)+(OR,NAND))([1,1])
        [[True, True], [False, False]]
        """
        print(self.defdef.find(func_name))
        return self.defdef.get(func_name)

    @overload
    def net(self,i,o,net:str | Callable) -> Callable:...
    @overload
    def net(self,net:str | Callable) -> Callable:...
    @overload
    def net(self,i,o_list,net_list) -> Callable:...
    def net(self,*args,print_=False,**kwargs):
        """
        >>> from monet import MoNetInitial,dict_slice
        >>> from monet.torch_ddf import torch_dict
        >>> m = MoNetInitial(dict_slice(torch_dict,0,3))
        >>> m.net(10,1,"fc")[0].name
        '@ddf:Linear(in_features=10, out_features=1, bias=True)'
        """
        return self.Layer(*args,**kwargs,get=self.__getattr__,print_=False)[1]

    def __getattr__(self, name):
        """call a function from defdef functionspace.
        >>> from monet import MoNetInitial
        >>> m = MoNetInitial()
        >>> from monet.example import pla2_Type
        >>> m.ddf(pla2_Type).name
        '@ddf:pla2_Type'
        >>> AND=m.pla2_AND
        >>> OR=m.pla2_OR
        >>> NAND=m.pla2_NAND
        >>> XOR =(AND+OR)*(NAND+OR)*NAND
        >>> XOR([1,1]),XOR([0,0]),XOR([0,1]),XOR([1,0])
        (True, True, False, False)
        >>> (m.f*(AND,(OR,NAND)))([1,1])
        [True, [True, False]]
        >>> (AND+(OR,NAND))([1,1])
        [[True, True], [False]]
        """
        try:
            return self.defdef.get(name)
        except Exception:
            if name not in self.funcspace and self.__funcspace__ is not None:
                func = eval(f"self.__funcspace__.{name}")
                if isinstance(func,Callable):
                    return self.ddf(name,func)
            else:
                raise AttributeError(f"{name} not found in funcspace")

    def __setattr__(self, name: str, value: Any) -> None:
        """call a function from defdef functionspace.
        >>> from monet import MoNetInitial
        >>> m = MoNetInitial()
        >>> from monet.example import pla2_Type
        >>> m.pla2_Type = pla2_Type
        >>> m.pla2_Type.name
        '@ddf:pla2_Type'
        """
        super().__setattr__(name,value)
        if isinstance(value,Callable) and not name.startswith("__"):
            if name not in ['defdef','Layer','f','find','ddf']:
                if name not in self.funcspace:
                    value = self.defdef.add(name,value)
                    self.__setattr__(name,value)

    def __getitem__(self,name):
        return self.__getattr__(name)

    def __setitem__(self,name,value):
        """call a function from defdef functionspace.
        >>> from monet import MoNetInitial
        >>> m = MoNetInitial()
        >>> from monet.example import pla2_Type
        >>> m['pla2_Type'] = pla2_Type
        >>> m['pla2_Type'].name
        '@ddf:pla2_Type'
        >>> m.pla2_Type.name
        '@ddf:pla2_Type'
        """
        assert isinstance(value,Callable), f"{value} is not a Callable"
        self.ddf(name,value)

    def __call__(self, arg1=None, arg2=None,*args) -> Any:
        """call a function from defdef functionspace.
        >>> from monet import MoNetInitial,torch_dict,pla2_Type
        >>> m = MoNetInitial(torch_dict)
        >>> m(pla2_Type).name
        '@ddf:pla2_Type'
        >>> m("fc").name
        '@ddf:fc'
        >>> m(10,[10,20,10],[["fc","act"]]).name
        {'seq': (10, [10, 20, 10], [['fc', 'act'], ['fc', 'act'], ['fc', 'act']])}
        """
        if arg1 is None:
            return self.f
        if isinstance(arg1,Callable):
            return self.ddf(arg1)
        if isinstance(arg1,str):
            if arg2 is None:
                return self.__getattr__(arg1)
            elif isinstance(arg2,Callable):
                return self.ddf(arg1,arg2)
        else:
            return self.net(arg1,arg2,*args)


def dict_slice(dictionary, start, stop):
    return {k: dictionary[k] for k in list(dictionary.keys())[start:stop]}

if __name__ == "__main__":
    import doctest
    doctest.testmod()