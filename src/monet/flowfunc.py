"""
This function defines the classes that enable any function to perform
multiplocation and division abilities
"""
from collections import OrderedDict
from copy import deepcopy
from typing import Any, Callable
from collections.abc import Sized
import inspect

try:
    from torch.nn import Module as Base
except Exception:
    Base = object

# Define a global variable IS_ PCALL, used to determine whether to call
# the pcall function to output the input-output values for each step
IS_PCALL = False

def get_name(func):
    """get the name of func"""
    # Obtain the representation of the function
    _fr = repr(func)
    # Get the class name of the function
    _fn = func.__class__.__name__
    if _fn in ['builtin_function_or_method','function','type']:
        _name = func.__name__
    elif _fn in _fr and not _fr.startswith(_fn):
        _name = _fn
    else:
        _name = _fr
    return _name

def seq(func_ord,*args,**kwargs):
    """Execute func_ord in sequence, with the output of the previous function as the
    input of the next function
    """
    x = [] if len(args) == 0 else [*args]
    x+= [] if len(kwargs) == 0 else [kwargs]
    if len(func_ord) > 0:
        if IS_PCALL:
            print("   ", end="")
        for n,(i,func) in enumerate(func_ord.items()):
            y = func(*x)
            if (mstr:=func.name) and IS_PCALL:
                print(f"   ({i}): "+mstr+f"\n    ┗━ {x} >>\n    ==", y )
            x = (y,)
        if len((y,))>0:
            return y
    return x

def loc_base(func_ord,*args,**kwargs):
    '''Execute func_ord in parallel, each function uses the same input, and finally
    concatenate all the outputs into a list
    '''
    x = [] if len(args) == 0 else [*args]
    x+= [] if len(kwargs) == 0 else [kwargs]
    all_y = []
    if len(func_ord) > 0:
        if IS_PCALL:
            print("   ", end="")
        for i,func in func_ord.items():
            y = func(*x)
            if (mstr:=func.name) and IS_PCALL:
                print(f"   ({i}): "+mstr+f"\n    ┗━ {x} >>\n    ==", y )
            all_y += [y]
        return all_y
    return x

def loc(func_ord,*args,**kwargs):
    '''Execute func_ord in parallel, each function uses the same input, and finally
    concatenate all the outputs into a list
    '''
    all_y = loc_base(func_ord,*args,**kwargs)
    for i in all_y:
        if not isinstance(i,Callable):
            return all_y
    return FuncModel(all_y,call="loc")


class FuncModel(Base): # type: ignore
    """
    >>> F = FuncModel(max)*FuncModel(abs)
    >>> F(-1,-2,-3,-4,-5)
    1
    >>> FuncModel()(1,{"a":2},b=3)
    [1, {'a': 2}, {'b': 3}]
    >>> (FuncModel()*max*abs)(-1,-2,-3,-4,-5)
    1
    >>> FuncModel()
    seq>Fn()
    >>> FuncModel(max).call.__name__
    'seq'
    """

    # Define an initializer function, with args as a list and call as a function or string
    def __init__(self, args=[], call: str | Callable = 'seq', name = ""):
        # Call the initializer function of the parent class
        super().__init__()
        # If args is not an iterable, convert it to a list
        args = [args] if not isinstance(args,Sized) else args
        # Initialize an ordered dictionary
        self._modules = OrderedDict()
        self.call = eval(call) if isinstance(call, str) else call
        self.name = name if name != "" else str(self.call)
        # Iterate over each element in args
        for i, arg in enumerate(args):
            # If arg is a Module or ddf object
            if self.is_ddf_funcmodel(arg):
                if self.is_funcmodel(arg.func):
                    self._modules[str(i)] = arg.func[0] if len(arg.func) == 1 else arg.func
                elif not self.is_ddf(arg):
                    if arg.name not in ['loc','seq']:
                        self._modules[str(i)+":"+arg.name] = arg[0] if len(arg) == 1 else arg
                        if len(arg) == 1:
                            self[i].__name__ = arg.name
                    else:
                        self._modules[str(i)] = arg[0] if len(arg) == 1 else arg
                else:
                    if arg.__name__ != str(get_name(arg.func)):
                        self._modules[str(i)+":"+arg.__name__] = arg
                    else:
                        self._modules[str(i)] = arg
            # If arg is not a Module or ddf object
            else:
                # Add it to the ordered dictionary
                if isinstance(arg, Sized):
                    self._modules[str(i)] = FuncModel(arg, call=call)
                else:
                    self._modules[str(i)] = self.ddf(arg)
        # Assign self.p to self.pcall
        self.p = self.pcall
        self.func = self.__call__
        self.id =f' *id:{id(self)}'

    def add_module(self, name, module):
        self._modules[name] = module

    def pcall(self,*args,**kwargs):
        # Define a global variable IS_PCALL to determine if pcall function is called
        global IS_PCALL
        # Set IS_PCALL to True
        IS_PCALL = True
        # Call the __call__ function of the parent class and assign the return value to output
        output = self.__call__(*args,**kwargs)
        # Set IS_PCALL to False
        IS_PCALL = False
        # Return output
        return output

    def __iter__(self):
        return iter(self._modules.values())

    def __getitem__(self,i):
        if isinstance(i,int):
            return list(self._modules.values())[i]
        else:
            return self._modules.__getitem__(i)

    def __repr__(self):
        # Get the __repr__ method of the parent class
        main_str = super().__repr__()
        # # If the __repr__ method of the parent class starts with the name of the current class, return the name of the current class and the __repr__ method of the parent class
        start_str = 'Fn'
        if main_str.startswith(start_str):
            return f"{get_name(self.call)}>{main_str}"
        # # Otherwise, return the name of the current class and the __repr__ method of the current class
        else:
            lines = [start_str + '(']
            # Iterate over the _modules attribute of the current class to get the __repr__ method of each module
            for key, func in self._modules.items():
                _rerp_str = []
                # Iterate over each module's __repr__ method and add indentation
                for i, r in enumerate(repr(func).split('\n')):
                    _rerp_str += ["  " + r] if i > 0 else [r]
                # Add each module's __repr__ method to the lines list
                lines += [f'  ({key}): ' + "\n".join(_rerp_str)]
            # Return the name of the current class, the __repr__ method of the current class, and the __repr__ method of each module
            main_str = f"{get_name(self.call)}>" + '\n'.join(lines)
            return main_str + '\n)' if len(lines) > 1 else main_str + ')'

    def __len__(self):
        return len(self._modules)

    def __add__(self,other):
        '''+ means loc call'''
        if isinstance(other,int):
            return FuncModel([self]+[FuncModel()]*other,call=loc,name=self.name+"x"+str(other))
        else:
            if len(self) == 1:
                nn_list = [*self] # not change
            elif self.call == loc:
                nn_list = [*self]
            else:
                nn_list = [self]

            if isinstance(other,Sized):
                if len(other) == 0:
                        nn_list += [other]
                elif self.is_ddf_funcmodel(other):
                    nn_list += [other]
                else:
                    nn_list += [FuncModel(other,call=loc,name=self.name)]
            else:
                nn_list += [other]
        return FuncModel(nn_list,call=loc,name=self.name)

    def __mul__(self,other):
        '''* means seq call'''
        if isinstance(other,int):
            return FuncModel([~self for i in range(other)],call=loc,name=self.name+"x"+str(other)) if other !=1 else ~self
        else:
            if len(self) == 1:
                nn_list = [*self]
            elif self.call == seq:
                nn_list = [*self]
            else:
                nn_list = [self]

            if isinstance(other,Sized):
                if len(other) == 0:
                    nn_list += [other]
                elif self.is_ddf_funcmodel(other):
                    nn_list += [other]
                else:
                    nn_list += [FuncModel(other,call=loc,name=self.name)]
            else:
                nn_list += [other]
        return FuncModel(nn_list,call=seq,name=self.name)

    def __pow__(self,other):
        '''** means seq call with deepcopy'''
        import copy
        if isinstance(other,int) and other > 0:
            return FuncModel([~self for i in range(other)],call=seq) if other !=1 else ~self
        else:
            raise ValueError('** must with a int number > 0')

    def __invert__(self):
        """
        >>> F = FuncModel(max)
        >>> (~F).id == F.id
        True
        """
        import copy
        return copy.deepcopy(self)

    def forward(self,*args,**kwargs):
        '''
        Call the call function and print out the parameters and output result of the function
        '''
        global IS_PCALL
        if IS_PCALL:
            print('@',self.__repr__().split('\n')[0],'>> ')
        output = self.call(self._modules,*args,**kwargs)
        if IS_PCALL:
            print(')>>', output)
        return output

    @staticmethod
    def ddf(func):
        return ddf(func)

    @staticmethod
    def is_ddf(func):
        return issubclass(func.__class__,(ddf))
    @staticmethod
    def is_funcmodel(func):
        return issubclass(func.__class__,(FuncModel))
    @staticmethod
    def is_ddf_funcmodel(func):
        return issubclass(func.__class__,(FuncModel,ddf))

    def __call__(self,*args, **kwargs) -> Any:
        if len(self._modules)==0 and isinstance(args[0],str) and len(args) == 1 and len(kwargs) == 0:
            return FuncModel(name=args[0])
        return self.forward(*args, **kwargs)


class ddf(FuncModel):
    '''Add the ability of flow passing and formulaic expression to any function
    >>> ddf(max).name
    '@ddf:max'
    >>> ddf(max)(1,2,3)
    3
    '''
    def __init__(self,func,name=None,initkwargs={}):
        super().__init__()
        self.func = func
        self.initkwargs = initkwargs
        self.__name__=str(get_name(self.func)) if name is None else str(name)
        if (str_func:=str(self.func)).startswith("<function <lambda>"):
            signature = inspect.signature(self.func)
            self.name = f'@{self.__class__.__name__}:'+f"{self.func.__qualname__}{signature}>"
        else:
            self.name=f'@{self.__class__.__name__}:'+str_func
        self.id =f' *id:{id(self)}'

    def forward(self, *args, **kwargs):
        if kwargs == {}:
            kwargs = self.initkwargs
        else:
            kwargs.update(self.initkwargs)
        res = self.func(*args, **kwargs)
        if isinstance(res,Callable):
            return ddf(res)
        return res

    def __repr__(self):
        self.func_id =f' *id:{id(self.func)}'
        return self.name + self.func_id

if __name__ == "__main__":
    import doctest
    doctest.testmod()
