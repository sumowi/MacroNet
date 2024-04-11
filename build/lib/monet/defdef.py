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
        "func":"funcA_B_C"
    }}
}

"""

# %%
# 函数输入参数的传递
# 函数输出值的传递
# 函数名的传递

from typing import OrderedDict, Callable, TypeVar, overload

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

    def __init__(self,funcspace={}, info_display=False):
        """
        >>> ddf = DefDefObj()
        >>> @ddf.add
        ... def logN(x, N=10):
        ...     "Define a logarithm with a base of N"
        ...     assert N > 0 and x > 0
        ...     print(f"def log{N}::logN")
        ...     return log(x, N)
        >>> ddf.find("log9")
        ('logN', {'N': 9})

        """
        self.funcspace = OrderedDict(funcspace)
        self.namespace = OrderedDict()
        self.info_display = info_display
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
    def add(self, dict_or_func: "namestr", dict={}, func={}) -> "dictkeys": ...
    def add(self, dict_or_func: "dict_func_type | namestr", *args):
        """add func from func or func_dict (see DefDefObj.funcspace_example)
        >>> ddf = DefDefObj()
        >>> ddf.add(ddf.parabolaA_B_C)
        dict_keys(['parabolaA_B_C'])
        >>> ddf.add(ddf.funcspace_example)
        dict_keys(['printTimes@File'])

        add func from dict or func_dict more than one
        >>> ddf = DefDefObj()
        >>> ddf.add([ddf.parabolaA_B_C,ddf.funcspace_example])
        [dict_keys(['parabolaA_B_C']), dict_keys(['printTimes@File'])]

        rename a func and rebuild with lambda
        >>> ddf = DefDefObj()
        >>> from math import log
        >>> ddf.add("logN", lambda x,N=10: log(x,N), "help info" )
        dict_keys(['logN'])
        """

        sth = dict_or_func
        if isinstance(sth, (dict, OrderedDict)):
            name = list(sth.keys())[0]
            child = sth.get(name)
            assert "func" in child, "func must define in k, but not found"
            assert isinstance(child["func"], Callable), "func must be callable"
            assert (
                len(sth) == 1
            ), "only one func can be added, you should ues list or tuple to add more at one time"
            if "help" not in child:
                child["help"] = child["func"].__doc__
            if "default" not in child:
                child["default"] = self.get_kwargs(child["func"])
            if "splits" not in child:
                child["splits"] = self.get_splits(name, child["default"])
            new_func = {name: child}
            self.funcspace.update(new_func)
            # self.funcfind(new_func)
            if self.info_display:
                return f"add '{name}' to funcspace"
            return new_func.keys()

        elif isinstance(sth, Callable):
            new_func = {
                sth.__name__: {
                    "default": self.get_kwargs(sth),
                    "help": sth.__doc__,
                    "func": sth,
                }
            }
            return self.add(new_func)

        elif isinstance(dict_or_func, (list, tuple)):
            add_list = []
            for func_dict in dict_or_func:
                add_result = self.add(func_dict)
                add_list += [add_result]
            return add_list

        elif isinstance(dict_or_func, str):
            if len(args) > 0:
                func = args[0]
                assert isinstance(func, Callable), f"{func} is not a Callable"
            else:
                raise ValueError("func must define in the second arg, but not found")
            help_doc = args[1:2] if args[1:2] != [] else func.__doc__
            new_func = {
                dict_or_func: {
                    "default": self.get_kwargs(func),
                    "help": help_doc,
                    "func": func,
                }
            }
            return self.add(new_func)

        else:
            raise ValueError(f"{dict_or_func} is not a list or tuple")

    def func_find(self, name):
        """find a func in funcspace
        >>> ddf = DefDefObj()
        >>> ddf.add(ddf.parabolaA_B_C)
        dict_keys(['parabolaA_B_C'])
        >>> ddf.func_find("parabolaC")
        False
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
                kwargs = self.get_namespace_kwargs(
                    k, child["default"], child["splits"], name
                )
                if kwargs is not False:
                    child["func"] = k
                    child["default"] = kwargs
                    self.namespace[name] = child
                    return k, self.namespace[name]["default"]
        # If the loop has not been found yet, return False
        return False

    @staticmethod
    def get_namespace_kwargs(
        funcspace_name, funcspace_kwargs, funcspace_splits, namespace_name
    ):
        """Empty or original characters indicate using default values, The value at the corresponding position must match the default value type
        >>> DefDefObj.get_namespace_kwargs("parabolaA_B_C",{"A":1,"B":2,"C":3},["parabola","_","_"],"parabola3_C")
        False
        >>> DefDefObj.get_namespace_kwargs("parabolaA_B_C",{"A":1,"B":2,"C":3,"D":4},["parabola","_","_"],"parabola3__5")
        {'A': 3, 'B': 2, 'C': 5, 'D': 4}
        >>> DefDefObj.get_namespace_kwargs("parabolaA_B_C",{"A":1,"B":2,"C":3},["parabola","_","_"],"parabola3_B")
        {'A': 3, 'B': 2, 'C': 3}
        >>> DefDefObj.get_namespace_kwargs("parabolaA_B_C",{"A":1,"B":2,"C":3},["parabola","_","_"],"parabola_3")
        {'A': 1, 'B': 3, 'C': 3}

        """
        kwargs = {}

        def setArgs(key, value):
            default = funcspace_kwargs.get(key)
            try:
                kwargs[key] = eval(value)
            except Exception:
                # If there is an error above, assign the value to a string or default value
                if key == value or value == "":
                    kwargs[key] = default
                else:
                    kwargs[key] = value
            finally:
                # The type of the new value must be the same as the default value type, except for those that default to None
                if type(kwargs[key]) == type(default) or default is None:
                    return True
                else:
                    return False
            return True

        if namespace_name.startswith(k := funcspace_splits[0]):
            a1, b1, c1 = funcspace_name.partition(k)
            a2, b2, c2 = namespace_name.partition(k)
            for k in funcspace_splits[1:]:
                funcspace_name = c1
                namespace_name = c2
                a1, b1, c1 = funcspace_name.partition(k)
                a2, b2, c2 = namespace_name.partition(k)
                if not setArgs(a1, a2):
                    return False
            if not setArgs(c1, c2):
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
        """
        splits = []
        for s in kwargs:
            if s in name:
                a, b, c = name.partition(s)
                assert a != "", "split key at start or near another key"
                splits.append(a)
                name = c
        return splits

    @staticmethod
    def get_kwargs(func):
        """Get the kwargs of a function
        >>> ddf = DefDefObj()
        >>> ddf.get_kwargs(ddf.parabolaA_B_C)
        {'A': 1, 'B': 2, 'C': 3}
        """
        import inspect

        try:
            kvs = inspect.signature(func).parameters
        except AttributeError as e:
            print(e)
            assert "get_kwargs Cannot be used on built-in functions"
        else:
            kwargs = {}
            func_name = func.__name__
            for k, v in kvs.items():
                if k in func_name:
                    kwargs[k] = v.default if "=" in str(v) else None
            return kwargs

    def get(self, func_name):
        """get a new func after add a func
        >>> ddf=DefDefObj()
        >>> ddf.add(ddf.parabolaA_B_C)
        dict_keys(['parabolaA_B_C'])
        >>> ddf.find("parabola1_2_3")
        ('parabolaA_B_C', {'A': 1, 'B': 2, 'C': 3})
        >>> ddf.get("parabola3_4_5")
        ddclass parabolaA_B_C({'A': 3, 'B': 4, 'C': 5})
        >>> ddf.get("parabola3_4_5")(1)
        12
        >>> ddf.get("parabola0_0_0")(1)
        0
        >>> ddf.get("parabola3")(1)
        8
        >>> ddf.get("parabola3")(1,A=0) # A=0 will not work
        8
        """
        if func_name in self.namespace:
            initkwargs = self.namespace[func_name]["default"]
            name = self.namespace[func_name]["func"]
        else:
            name, initkwargs = self.func_find(func_name)

        func = self.funcspace[name]["func"]

        class ddclass(object):
            def __init__(self, func, initkwargs) -> None:
                self.func = func
                self.initkwargs = initkwargs
                pass

            def __repr__(self) -> str:
                return f"ddclass {name}({self.initkwargs})"

            def __call__(self, *args, **kwargs):
                if kwargs == {}:
                    kwargs = self.initkwargs
                else:
                    kwargs.update(self.initkwargs)
                return self.func(*args, **kwargs)

        return ddclass(func, initkwargs)

    # def __getattr__(self, name: str) -> Any:
    #     """get a new func after add a func
    #     """
    #     if name in self.namespace:
    #         return self.namespace[name]
    #     else:
    #         return self.get(name)

    def __call__(self, func, call=None):
        """get a new func after add a func
        >>> ddf=DefDefObj()
        >>> ddf(ddf.parabolaA_B_C)
        dict_keys(['parabolaA_B_C'])
        >>> ddf.find("parabola1_2_3")
        ('parabolaA_B_C', {'A': 1, 'B': 2, 'C': 3})
        >>> ddf("parabola3_4_5")
        ddclass parabolaA_B_C({'A': 3, 'B': 4, 'C': 5})
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
        if isinstance(func, str):
            return self.add(func, call)

        return self.get(func)

    def get_args(net="fc_1"):
        # _分隔值形参数，.分割字符串形式参数
        # .在_前面,只能传递一个参数,会在参数名称里
        args = net.split("_")
        args_str = args[0].split(".")
        name = args_str[0]
        # 先添加字符串形式参数
        args_str = args_str[1:]
        # 从name的最后一个字符提取维度值
        args_opt = [eval(name[-1])] if name[-1] in "1234567890" else []
        name = name[:-1] if name[-1] in "1234567890" else name
        # 添加值形式从参数,''返回空值
        args_opt += [eval(i) if i != "" else [] for i in args[1:]]
        return name, args_str, args_opt


# %%
if __name__ == "__main__":
    import doctest

    doctest.testmod()


# %%


# kljhlk
