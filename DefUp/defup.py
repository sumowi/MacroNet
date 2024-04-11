from collections import OrderedDict

try:
    import torch
    import torch.nn as nn
    Module_base = nn.Module
except:
    Module_base = None


def get_name(func):
    fr=repr(func)
    fn=func.__class__.__name__
    if fr.startswith(fn):
        return fr
    elif fn in ['builtin_function_or_method','function']:
        return func.__name__
    else:
        return fn


def SEQ(ord,*args,**kwargs):
    y = x = (args, kwargs)
    print(f"input =", x)
    for i,func in enumerate(ord.items()):
        y = func(*x)
        print(f"  {i} : {get_name(func)}({x}) =", y )
        x = y
    return y

def LIC(ord,*args,**kwargs):
    y = x = (args, kwargs)
    print(f"input =", x)
    all_y = [y]
    for i,func in enumerate(ord.items()):
        y = func(*x)
        print(f"  {i} : {get_name(func)}({x}) =", y )
        x = y
        all_y+=[y]
    return y if len(all_y)==1 else all_y



def Base(base_cls: list = [Module_base] ,call = SEQ ):


    class Module(*base_cls): # type: ignore

        def __init__(self,args=[],call=call):
            super().__init__()
            args = [args] if not hasattr(args,'__iter__') else args
            self._modules=OrderedDict()
            for i,arg in enumerate(args):
                if arg.__class__.__name__ in ("Module","dup"):
                    self._modules[str(i)] = arg
                else:
                    self._modules[str(i)] = self.dup(arg)
            self.call = call

        def __call__(self,*args,**kwargs):
            if len(self._modules)==0:
                res = []
                res += [args] if len(args)>0 else []
                res += [kwargs] if len(kwargs)>0 else []
                return res[0] if len(res)==1 else res
            else:
                return self.call(self,*args,**kwargs)

        def __iter__(self):
            return iter(self._modules.values())

        def __getitem__(self,i):
            return self._modules.__getitem__(i)

        def  __repr__(self):
            main_str = super().__repr__()
            if main_str.startswith(self.__class__.__name__):
                return f"{get_name(self.call)}>"+main_str
            else:
                lines = [self.__class__.__name__+'(']
                for key,func in self._modules.items():
                    _rerp_str = []
                    for i,r in enumerate(repr(func).split('\n')):
                        _rerp_str+=["  "+r] if i>0 else [r]
                    lines+=[f'  ({key}): ' + "\n".join(_rerp_str)]
                return f"{get_name(self.call)}>"+'\n'.join(lines+[')'])

        def __len__(self):
            return len(self._modules)

        def __add__(self,other):
            nn_list = [*self] if self.call == LIC else [self]
            nn_list += [*other] if self.is_dup(other) and other.call != SEQ else [other]
            return Module(nn_list,call=LIC)

        def __mul__(self,other):
            nn_list = [*self] if self.call == SEQ else [self]
            nn_list += [*other] if self.is_dup(other) and other.call != LIC else [other]
            return Module(nn_list,call=SEQ)

        def forward(self,*args,**kwargs):
            return self.__call__(*args,**kwargs)

        @staticmethod
        def dup(func):
            return dup(func)

        @staticmethod
        def is_dup(func):
            return func.__class__.__name__ in ("Module","dup")  # type: ignore

    class dup(Module):
        def __init__(self,func):
            super().__init__()
            self._modules = OrderedDict([('0',func)])
            self.__call__ = func.__call__

        def __repr__(self):
            main_str=self.__class__.__name__
            return f'@{main_str}:'+get_name(self._modules['0'])

    return Module

M =  Base(base_cls=[Module_base])
m =  Base(base_cls=[])

X =  M()
x =  m()

class AddBias(M):
    def __init__(self):
        super(AddBias, self).__init__()
        self.bias = nn.Parameter(torch.FloatTensor(1))

    def forward(self, x):
        return x + self.bias
