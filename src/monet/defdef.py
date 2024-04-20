"""
this is a module ti define a function how to define.

DefDefObj.funcspace={
    "funcA_B_C":{
        "default":{
            "A":1,
            "B":2,
            "C":3
        },
        ”help":"xxxxxx",
        ”splits":["func","_","_"],
        "func":lambda x,A=1,B=2,C=3: A*x*x+B*x+C
    }}
}

DefDefObj.namespace={
    "func1_2_3":{
        "default":{
            "A":1,
            "B":2,
            "C":3
        },
        ”help":"xxxxxx",
        ”splits":["func","_","_"],
        "func":"funcA_B_C"
    }}
    "func3_4_5":{
        "default":{
            "A":3,
            "B":4,
            "C":5
        },
        ”help";"xxxxxx",
        ”splits":["func","_","_"]
t66tt6666666l,        "func":"funcA_B_C"
    }}
}

"""

# %%
# 函数输入参数的传递
# 函数输出值的传递
# 函数名的传递

from typing import OrderedDict, Callable, TypeVar, overload
from monet.flowfunc import FuncModel,ddf
from monet._monet import get_args
dict_func_type = TypeVar(
    "dict_func_type", dict, Callable, list[dict | Callable], tuple[list | Callable]
)
dictkeys = TypeVar("dictkeys", bound=OrderedDict)
namestr = TypeVar("namestr", bound=str)


class DefDefObj:
    funcspace_example = {
        "printTimes@File": {
            "default": {"Times": 3, "File": "out.txt"},
            "help": "Print a certain content N times into a file",
            "splits": ["print", "@"],
            "func": lambda *args, Times=3, File="out.txt", **kwargs: (
                print(*args, **kwargs, file=open(File, "a+")) for i in range(Times)
            ),
        }
    }

    def __init__(self,spaceobj=None):
        """
        >>> ddf = DefDefObj()
        >>> @ddf.add
        ... def logN(x, N=10):
        ...     "Define a logarithm with a base of N"
        ...     assert N > 0 and x > 0
        ...     print(f"def log{N}::logN")
        ...     return log(x, N)
        >>> logN.name
        '@ddf:logN'
        >>> ddf.find("log9")
        ('logN', {'N': 9})
        >>> ddf.add(logN).name
        '@ddf:logN'
        """
        self.funcspace = spaceobj.funcspace if spaceobj is not None else OrderedDict()
        self.namespace = spaceobj.namespace if spaceobj is not None else OrderedDict()

        self.find = self.func_find

    @staticmethod
    def parabolaA_B_C(x, A=1, B=2, C=3):
        "Define a parabola: y=A*x*x+B*x+C"
        return A * x * x + B * x + C

    @overload
    def add(self, dict_or_func: "dict | Callable") -> "dictkeys": ...
    @overload
    def add(self, dict_or_func: list | tuple) -> "list[dictkeys]": ...
    @overload
    def add(self, dict_or_func: "namestr", call:"Callable") -> "dictkeys": ...
    def add(self, dict_or_func: "dict_func_type | namestr", *args):
        """add func from func or func_dict (see DefDefObj.funcspace_example)
        >>> ddf = DefDefObj()
        >>> ddf.add(ddf.parabolaA_B_C).name
        '@ddf:parabolaA_B_C'
        >>> ddf.add(ddf.funcspace_example).name
        '@ddf:printTimes@File'

        add func from dict or func_dict more than one
        >>> ddf.add([ddf.parabolaA_B_C,ddf.funcspace_example])
        ['@ddf:parabolaA_B_C', '@ddf:printTimes@File']

        rename a func and rebuild with lambda
        >>> from math import log
        >>> ddf.add("logN", lambda x,N=10: log(x,N), "help info" ).name
        '@ddf:logN'
        >>> ddf.funcspace['logN']['default']
        {'N': 10}

        >>> f = ddf.add(ddf.parabolaA_B_C)
        >>> ddf.add(f).name
        '@ddf:parabolaA_B_C'
        """

        sth = dict_or_func
        if isinstance(sth, (dict, OrderedDict)):
            if len(sth) == 1:
                name = list(sth.keys())[0]
                child = sth.get(name)
                if isinstance(child,Callable):
                    return self.add(name,child)
                assert "func" in child, "func must define in k, but not found"
                assert isinstance(child["func"], Callable), "func must be callable"
                if "help" not in child:
                    child["help"] = child["func"].__doc__
                if "default" not in child:
                    child["default"] = self.get_kwargs(child["func"],name = name)
                if "splits" not in child:
                    child["splits"] = self.get_splits(name, child["default"])
                new_func = {name: child}
                self.funcspace.update(new_func)
                # self.funcfind(new_func)
                return ddf(new_func[name]['func'],name,child['default'])
            else:
                return self.add([{k:v} for k,v in dict_or_func.items()])

        elif isinstance(sth, Callable):
            if ddf.is_ddf(sth):
                return self.add(sth.func)
            initw = self.get_kwargs(sth)
            new_func = {
                sth.__name__: {
                    "default": initw,
                    "help": sth.__doc__,
                    "func": sth,
                }
            }
            self.add(new_func)
            return ddf(sth,sth.__name__,initw)

        elif isinstance(dict_or_func, (list, tuple)):
            add_list = []
            for func_dict in dict_or_func:
                add_result = self.add(func_dict)
                add_list += [add_result.name]
            return add_list

        elif isinstance(dict_or_func, str):
            if len(args) > 0:
                call = args[0]
                assert isinstance(call, Callable), f"{call} is not a Callable"
            else:
                def add_func(call):
                    assert isinstance(call, Callable), f"{call} is not a Callable or is None"
                    return self.add(dict_or_func, call)

                return add_func

            if ddf.is_ddf(call):
                return self.add(dict_or_func,call.func)

            help_doc = args[1:2] if args[1:2] != [] else call.__doc__
            initw=self.get_kwargs(call,dict_or_func)
            new_func = {
                dict_or_func: {
                    "default": self.get_kwargs(call,dict_or_func),
                    "help": help_doc,
                    "func": call,
                }
            }
            self.add(new_func)
            return ddf(call,dict_or_func,initw)

        else:
            raise ValueError(dict_or_func,*args)

    def func_find(self, name):
        """find a func in funcspace
        >>> ddf = DefDefObj()
        >>> ddf.add(ddf.parabolaA_B_C).name
        '@ddf:parabolaA_B_C'
        >>> ddf.func_find("parabolaA_B_C")
        ('parabolaA_B_C', {'A': 1, 'B': 2, 'C': 3})
        >>> ddf.func_find("parabola3_4_5")
        ('parabolaA_B_C', {'A': 3, 'B': 4, 'C': 5})
        >>> ddf.func_find("parabola3")
        ('parabolaA_B_C', {'A': 3, 'B': 2, 'C': 3})
        >>> ddf.func_find("parabola_4_5")
        ('parabolaA_B_C', {'A': 1, 'B': 4, 'C': 5})
        """

        import copy

        for k, v in self.funcspace.items():
            child = copy.deepcopy(v)
            if name == k:
                child["func"] = k
                self.namespace[name] = child
                return k, self.namespace[name]["default"]
            else:
                kwargs = self.get_namespace_kwargs( child["default"], child["splits"], name
                )
                if kwargs is False:
                    if name[-1] != '.':
                        kwargs = self.get_namespace_kwargs( child["default"], child["splits"], name+'.'
                    )
                    if kwargs is False:
                        kwargs = self.get_namespace_kwargs( child["default"], child["splits"], name+'_'
                    )
                if kwargs is not False:
                    child["func"] = k
                    child["default"] = kwargs
                    self.namespace[name] = child
                    return k, self.namespace[name]["default"]
        # If the loop has not been found yet, return False
        raise ValueError(f"{name} no found, you should defdef it first.")

    @staticmethod
    def get_namespace_kwargs(funcspace_kwargs, funcspace_splits, namespace_name
    ):
        """Empty or original characters indicate using default values, The value at the corresponding position must match the default value type
        >>> DefDefObj.get_namespace_kwargs({"A":1,"B":2,"C":3},["parabola","_","_"],"parabola3_C")
        False
        >>> DefDefObj.get_namespace_kwargs({"A":1,"B":2,"C":3,"D":4},["parabola","_","_"],"parabola3__5")
        {'A': 3, 'B': 2, 'C': 5, 'D': 4}
        >>> DefDefObj.get_namespace_kwargs({"A":1,"B":2,"C":3},["parabola","_","_"],"parabola3_B")
        {'A': 3, 'B': 2, 'C': 3}
        >>> DefDefObj.get_namespace_kwargs({"A":1,"B":2,"C":3},["parabola","_","_"],"parabola_3")
        {'A': 1, 'B': 3, 'C': 3}
        >>> DefDefObj.get_namespace_kwargs({"A":1,"B":2,"C":3},["parabola","_","_"],"parabola_3")
        {'A': 1, 'B': 3, 'C': 3}
        >>> from monet import MoNetInitial, pla2_Type
        >>> m = MoNetInitial()
        >>> m.ddf(pla2_Type).name
        '@ddf:pla2_Type'
        >>> m["pla2_AND"].initkwargs
        {'Type': 'AND'}
        """
        kwargs = {}

        def setArgs(key, value):
            if key=='':
                return True
            default = funcspace_kwargs.get(key)
            try:
                value = eval(value)
                if isinstance(default,bool):
                    value = bool(value)
                kwargs[key] = value
            except Exception:
                # If there is an error above, assign the value to a string or default value
                if key == value or value == "" or value is None:
                    kwargs[key] = default
                else:
                    kwargs[key] = value
            finally:
                # The type of the new value must be the same as the default value type, except for those that default to None
                if type(kwargs[key]) == type(default) or default is None:
                    return True
                else:
                    return False

        if namespace_name.startswith(k := funcspace_splits[0]):
            a, b, c = namespace_name.partition(k)
            key_list=list(funcspace_kwargs.keys())
            n=0
            for k in funcspace_splits[1:]:
                namespace_name = c
                a, b, c = namespace_name.partition(k)
                if not setArgs(key_list[n], a):
                    return False
                n+=1
            if not setArgs(key_list[n], c):
                return False
            # If the segmentation rules match, return kwargs
            funcspace_kwargs.update(kwargs)
            kwargs = funcspace_kwargs
            return kwargs
        # If the segmentation rules do not match, return False
        return False

    @staticmethod
    def get_splits(name, kwargs):
        """Only corresponds to one position and cannot be at the forefront.
        Two keys cannot be directly adjacent.
        >>> DefDefObj.get_splits("printTimes@File",{"Times":3,"File":"out.txt","More":"more.txt"})
        ['print', '@']
        >>> DefDefObj.get_splits("printTimes@File.txt",{"Times":3,"File":"out.txt","More":"more.txt"})
        ['print', '@', '.txt']
        >>> DefDefObj.get_splits("parabola1_2_3",{"A":1,"B":2,"C":3})
        ['parabola', '_', '_']
        """
        splits = []
        for k,v in kwargs.items():
            s = k
            if s not in name and str(v) in name:
                s = str(v)
            if s in name:
                a, b, c = name.partition(s)
                assert a != "", "split key at start or near another key"
                splits.append(a)
                name = c
        if name!="":
            splits.append(name)
        return splits

    @staticmethod
    def get_kwargs(func,name=None):
        """Get the kwargs of a function
        >>> ddf = DefDefObj()
        >>> ddf.get_kwargs(ddf.parabolaA_B_C)
        {'A': 1, 'B': 2, 'C': 3}
        >>> ddf.get_kwargs(ddf.parabolaA_B_C,name="parabola3_4_5")
        {'A': 3, 'B': 4, 'C': 5}
        """
        import inspect

        try:
            kvs = inspect.signature(func).parameters
        except AttributeError as e:
            print(e)
            assert "get_kwargs Cannot be used on built-in functions"
        else:
            kwargs = {}
            func_name = func.__name__ if name is None else name
            for k, v in kvs.items():
                if k in func_name:
                    kwargs[k] = v.default if "=" in str(v) else None
            if kwargs == {}:
                name,num,str_,arg_ = get_args(func_name)
                args=[*num,*str_,*arg_]
                for i in range(n:=min(len(args),len(kvs))):
                    kwargs[list(kvs.keys())[len(kvs)-n+i]]=args[len(args)-n+i]
            return kwargs

    def get(self, func_name=None):
        """get a new func after add a func
        >>> ddf=DefDefObj()
        >>> ddf.add(ddf.parabolaA_B_C).name
        '@ddf:parabolaA_B_C'
        >>> ddf.find("parabola1_2_3")
        ('parabolaA_B_C', {'A': 1, 'B': 2, 'C': 3})
        >>> ddf.get("parabola3_4_5").name
        '@ddf:parabola3_4_5'
        >>> ddf.get("parabola3_4_5")(1)
        12
        >>> ddf.get("parabola0_0_0")(1)
        0
        >>> ddf.get("parabola3")(1)
        8
        >>> ddf.get("parabola3")(1,A=0) # A=0 will not work
        8
        >>> ddf.get()
        seq>Fn()
        """
        import inspect
        if func_name is None:
            return FuncModel(defdef=self)
        if func_name in self.namespace:
            initkwargs = self.namespace[func_name]["default"]
            name = self.namespace[func_name]["func"]
        else:
            name, initkwargs = self.func_find(func_name)

        func = self.funcspace[name]["func"]
        if list(inspect.signature(func).parameters.keys()) == list(initkwargs.keys()):
            return ddf(func(**initkwargs), func_name)
        else:
            return ddf(func, func_name,initkwargs)

    def __call__(self, func, call=None):
        """get a new func after add a func
        >>> ddf=DefDefObj()
        >>> ddf(ddf.parabolaA_B_C).name
        '@ddf:parabolaA_B_C'
        >>> ddf.find("parabola1_2_3")
        ('parabolaA_B_C', {'A': 1, 'B': 2, 'C': 3})
        >>> ddf("parabola3_4_5").name
        '@ddf:parabola3_4_5'
        >>> ddf("parabola3_4_5")(1)
        12
        >>> ddf("parabola0_0_0")(1)
        0
        >>> ddf("parabola3")(1)
        8
        >>> ddf("parabola3")(1,A=0) # A=0 will not work
        8
        """
        if isinstance(func, Callable):
            return self.add(func)
        return self.get(func)

# %%
if __name__ == "__main__":
    import doctest

    doctest.testmod()


# %%


# kljhlk
