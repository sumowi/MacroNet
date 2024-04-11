class MoNetInitial:
    def __init__(self,funcspace={}) -> None:
        """f is an blank func, which return the input.
        >>> from monet import MoNetInitial
        >>> m = MoNetInitial()
        >>> m.f
        seq>Fn()
        >>> m.f(1,2,3)
        [1, 2, 3]
        """
        from monet.defdef import DefDefObj
        from monet._monet import Layer
        from typing import OrderedDict

        self.funcspace = OrderedDict()
        self.namespace = OrderedDict()
        self.defdef = DefDefObj(spaceobj=self)
        self.defdef.add(funcspace)
        self.f = self.defdef.get()
        self.find = self.defdef.find
        self.Layer = Layer

    def ddf(self,*args,**kwargs):
        """defdef a function,which return a callable function
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
        """
        return self.defdef.add(*args,**kwargs)

    def fit(self,*args,**kwargs):
        """call a function from defdef functionspace.
        >>> from monet import MoNetInitial
        >>> m = MoNetInitial()
        >>> from monet.example import pla2_Type,func_pla
        >>> m.ddf(pla2_Type).name
        '@ddf:pla2_Type'
        >>> AND=m.fit("pla2_AND")
        >>> OR=m.fit("pla2_OR")
        >>> NAND=m.fit("pla2_NAND")
        >>> XOR =(AND+OR)*(NAND+OR)*NAND
        >>> XOR([1,1]),XOR([0,0]),XOR([0,1]),XOR([1,0])
        (True, True, False, False)
        >>> (m.f*(AND,(OR,NAND)))([1,1])
        [True, [True, False]]
        >>> (AND+(OR,NAND))([1,1])
        [True, True, False]
        """
        return self.defdef.get(*args,**kwargs)

    def net(self,*args,print_=False,**kwargs):
        """
        >>> from monet import MoNetInitial,dict_slice
        >>> from monet.torch_ddf import torch_dict
        >>> m = MoNetInitial(dict_slice(torch_dict,0,3))
        >>> m.net(10,1,"fc")[0].Net
        Linear(in_features=10, out_features=1, bias=True)
        """
        return self.Layer(*args,**kwargs,get=self.fit,print_=False)[1]

def dict_slice(dictionary, start, stop):
    return {k: dictionary[k] for k in list(dictionary.keys())[start:stop]}

if __name__ == "__main__":
    import doctest
    doctest.testmod()