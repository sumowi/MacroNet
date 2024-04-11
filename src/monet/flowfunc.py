"""
This function defines the classes that enable any function to perform
multiplication and division abilities
"""
from collections import OrderedDict
from typing import Any, Callable
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
        for i,func in func_ord.items():
            y = func(*x)
            # 如果mstr不等于')'且IS_PCALL为真，则打印出i，mstr，x和y
            if (mstr:=get_name(func).split('\n')[-1]) != ')' and IS_PCALL:
                print(f" ({i}): "+mstr+f"\n  $ {x} >>\n  ==", y )
            # 将y赋值给x
            x = (y,)
        # 如果y的长度大于0，则返回y
        if len((y,))>0:
            return y
    return x

def lic(func_ord,*args,**kwargs):
    '''Execute func_ord in parallel, each function uses the same input, and finally
    concatenate all the outputs into a list
    '''
    x = [] if len(args) == 0 else [*args]
    x+= [] if len(kwargs) == 0 else [kwargs]
    all_y = []
    if len(func_ord) > 0:
        for i,func in func_ord.items():
            y = func(*x)
            if (mstr:=get_name(func).split('\n')[-1]) != ')' and IS_PCALL:
                print(f" ({i}): "+mstr+f"\n  $ {x} >>\n  ==", y )
            all_y += [y]
        if len((all_y,))>0 :
            return all_y
    return x

def cat(func_ord,*args,**kwargs):
    '''Execute func_ord in parallel, each function uses the same input, and finally
    concatenate all the outputs into a list
    '''
    x = [] if len(args) == 0 else [*args]
    x+= [] if len(kwargs) == 0 else [kwargs]
    all_y = []
    if len(func_ord) > 0:
        for i,func in func_ord.items():
            y = func(*x)
            if (mstr:=get_name(func).split('\n')[-1]) != ')' and IS_PCALL:
                print(f" ({i}): "+mstr+f"\n  $ {x} >>\n  ==", y )
            all_y += [*y] if hasattr(y,'__iter__') else [y]
        if len((all_y,))>0 :
            return all_y
    return x

class FuncModel: # type: ignore
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
        args = [args] if not hasattr(args, '__iter__') else args
        # Initialize an ordered dictionary
        self._modules = OrderedDict()
        self.call = eval(call) if isinstance(call, str) else call
        # Iterate over each element in args
        for i, arg in enumerate(args):
            # If arg is a Module or ddf object
            if self.is_ddf(arg):
                # Add the first element of arg to the ordered dictionary, if arg has only one element and its representation has more than one line, add the first element of arg to the ordered dictionary, otherwise add arg to the ordered dictionary
                self._modules[str(i)] = [*arg][0] if repr(arg).count('\n') != 0 and len(arg) == 1 else arg
            # If arg is not a Module or ddf object
            else:
                # Copy arg and add it to the ordered dictionary
                if hasattr(arg, '__len__'):
                    self._modules[str(i)] = FuncModel(arg,lic)
                else:
                    self._modules[str(i)] = self.ddf(arg)
        # Assign self.p to self.pcall
        self.p = self.pcall
        self.func = self
        self.name = name if name != "" else str(call)
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
        # If the __repr__ method of the parent class starts with the name of the current class, return the name of the current class and the __repr__ method of the parent class
        start_str = 'Fn'
        if main_str.startswith(start_str):
            return f"{get_name(self.call)}>{main_str}"
        # Otherwise, return the name of the current class and the __repr__ method of the current class
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
        '''+ means seq call'''
        if isinstance(other,int):
            return self+[FuncModel()]*other if other !=1 else self+[FuncModel()]
        else:
            if len(self) == 1:
                nn_list = [self]
            elif self.call == cat:
                nn_list = [*self]
            else:
                nn_list = [self]

            if hasattr(other,'__len__'):
                if len(other) == 1:
                    nn_list += [other]
                elif self.is_ddf(other):
                    if other.call == lic:
                        nn_list += [*other]
                    else:
                        nn_list += [other]
                else:
                    nn_list += [FuncModel(other,call=lic)]
            else:
                nn_list += [other]
        return FuncModel(nn_list,call=cat)

    def __and__(self,other):
        """ & means lic call with deepcopy"""
        import copy
        if isinstance(other,int):
            return FuncModel([copy.deepcopy(self) for i in range(other)],call=lic)+1 if other !=1 else copy.deepcopy(self)+1
        else:
            return copy.deepcopy(self)*other+other

    def __mul__(self,other):
        '''* means seq call'''
        if isinstance(other,int):
            return FuncModel([self]*other) if other !=1 else self
        else:
            if len(self) == 1:
                nn_list = [self]
            elif self.call == seq:
                nn_list = [*self]
            else:
                nn_list = [self]

            if hasattr(other,'__len__'):
                if len(other) == 1:
                    nn_list += [other]
                elif self.is_ddf(other):
                    if other.call == seq:
                        nn_list += [*other]
                    else:
                        nn_list += [other]
                else:
                    nn_list += [FuncModel(other,call=lic)]
            else:
                nn_list += [other]
        return FuncModel(nn_list,call=seq)

    def __pow__(self,other):
        '''*** means seq call with deepcopy'''
        import copy
        if isinstance(other,int):
            return FuncModel([copy.deepcopy(self) for i in range(other)],call=seq) if other !=1 else copy.deepcopy(self)
        else:
            return copy.deepcopy(self)*other

    def forward(self,*args,**kwargs):
        '''
        Call the call function and print out the parameters and output result of the function
        '''
        global IS_PCALL
        if IS_PCALL:
            print('@',self.__repr__().split('\n')[0],'>>\n   ', *args, **kwargs)
        output = self.call(self._modules,*args,**kwargs)
        if IS_PCALL:
            print(')>>', output)
        return output

    @staticmethod
    def ddf(func):
        return ddf(func)

    @staticmethod
    def is_ddf(func):
        return issubclass(func.__class__,(ddf))   # type: ignore

    def __call__(self, *args, **kwargs) -> Any:
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

        self._modules = OrderedDict([('0',func)])
        self.func = func
        self.initkwargs = initkwargs
        self.__name__=str(get_name(self._modules['0'])) if name is None else str(name)
        self.name=f'@{self.__class__.__name__}:'+self.__name__
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
        return self.name + self.id

if __name__ == "__main__":
    import doctest
    doctest.testmod()
