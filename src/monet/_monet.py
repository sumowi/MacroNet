"""
>>> _ = Layer(10,[10,20,10],[["fc","act"]])
10 seq [10, 20, 10] [['fc', 'act'], ['fc', 'act'], ['fc', 'act']]
┗━ 10 seq [10, 10] ['fc', 'act']
   ┗━ 10 -> net : 10 fc
   ┗━ 10 -> net : 10 act
   -> 10
┗━ 10 seq [20, 20] ['fc', 'act']
   ┗━ 10 -> net : 20 fc
   ┗━ 20 -> net : 20 act
   -> 20
┗━ 20 seq [10, 10] ['fc', 'act']
   ┗━ 20 -> net : 10 fc
   ┗━ 10 -> net : 10 act
   -> 10
-> 10
"""

from typing import Callable, Sized
from monet.flowfunc import FuncModel as Fn
from monet.flowfunc import ddf
import inspect
def get_name_num(_str="fc231"):
    """
    >>> get_name_num("fc231")
    ('fc', [231])
    >>> get_name_num("fc23_1")
    ('fc23_', [1])
    """
    str_=_str
    num_=""
    while str_[-1] in "1234567890":
        num_ =str_[-1]+num_
        str_=str_[:-1]
    if num_ !="":
        num_=[eval(num_)]
    else:
        num_=[]
    return str_,num_

def get_args(net="fc_1"):
    """
    >>> get_args("fc_1")
    ('fc', [], [], [1])
    >>> get_args("cv2_1_2")
    ('cv', [2], [], [1, 2])
    >>> get_args("act.Relu_()")
    ('act', [], ['Relu'], [()])
    >>> get_args("cv2__2")
    ('cv', [2], [], ['', 2])
    >>> get_args("aap_(6,6)")
    ('aap', [], [], [(6, 6)])
    """
    args=net.split("_")
    args_str = args[0].split(".")
    name_num = args_str[0]
    name,num=get_name_num(name_num)
    _args=[]
    for i in args[1:]:
        try:
            _args.append(eval(i))
        except Exception:
            _args.append(i)
    args_str=args_str[1:]
    return name,num,args_str,_args

def mn_get(func_name):
    """
    >>> mn_get("fc")(10,1).name
    '@ddf:Linear(in_features=10, out_features=1, bias=True)'
    >>> mn_get("fc_0")(10,1).name
    '@ddf:Linear(in_features=10, out_features=1, bias=False)'
    >>> mn_get("bfc")((10,20),1).name
    '@ddf:Bilinear(in1_features=10, in2_features=20, out_features=1, bias=True)'
    """
    from monet.torch_ddf import torch_dict
    name,num,args_str,args = get_args(func_name)
    for k,func in torch_dict.items():
        _name, _num, _args_str, _args = get_args(k)
        if name == _name:
            if num is []:
                num = _num
            if args_str==[]:
                args_str = _args_str
            else:
                for i,v in enumerate(args_str):
                    if v == "":
                        args_str[i] = _args_str[i]

            if args==[]:
                args = _args
                # print(_args)
            else:
                for i,v in enumerate(args):
                    if v == "":
                        args[i] = _args[i]
            _args = [*num,*args_str,*args]
            # print(_args)
            import inspect
            if len(list(inspect.signature(func).parameters.keys())) == len(_args):
                return ddf(func(*_args))
            else:
                return ddf(lambda *args,**kwargs: func(*args,*_args,**kwargs))


class monet(ddf):
    def __init__(self,module,i=-1,o=0,net='',defdef=None,in_dim=-1):
        super().__init__(module)
        self.i = i
        self.auto_i = (i == 0)
        self.o = o
        self.net = net
        self.defdef = defdef
        self.mn_get = mn_get if defdef is None else defdef.get
        self.in_dim = 1 if net.startswith("cv") and in_dim==-1 else in_dim
        if Fn.is_ddf(module):
            self.initkwargs = module.initkwargs
            self.name = module.name

    def forward(self,x,*args,**kwargs):
        if self.auto_i is True and self.i != x.shape[self.in_dim]:
            print(f"[Info] Input size={self.i} is not match, reset to {x.shape[self.in_dim]}:")
            self.i = x.shape[self.in_dim]
            Nets = Layer(self.i,self.o,self.net,self.in_dim,self.defdef)
            super().__init__(Nets)
            if hasattr(x,"device") and hasattr(self.func,"to"):
                self.func.to(x.device)
        return super().forward(x,*args,**kwargs)


def Layer(i: int | str | list | tuple=0,
          o:int | list[int]=1,
          net:str | list[str,Callable] ="fc_1",
          in_dim=-1,
          defdef = None,
          gap=0,
          print_=True):
    """
    >>> _ = Layer("fc_True")
    0 seq [1] ['fc_True']
    ┗━ 0 -> net : 1 fc_True
    -> 1
    >>> _ = Layer(10,1,"fc")
    10 seq [1] ['fc']
    ┗━ 10 -> net : 1 fc
    -> 1
    >>> _ = Layer(10,[20,10],"fc")
    10 seq [20, 10] ['fc', 'fc']
    ┗━ 10 -> net : 20 fc
    ┗━ 20 -> net : 10 fc
    -> 10
    >>> _ = Layer(10,[20],["fc","act"])
    10 seq [20, 20] ['fc', 'act']
    ┗━ 10 -> net : 20 fc
    ┗━ 20 -> net : 20 act
    -> 20
    >>> _ = Layer(10,(10,20),"fc")
    10 loc (10, 20) ('fc', 'fc')
    ┗━ 10 -> net : 10 fc
    ┗━ 10 -> net : 20 fc
    -> (10, 20)
    >>> _ = Layer(10,20,("fc","fc"))
    10 loc (20, 20) ('fc', 'fc')
    ┗━ 10 -> net : 20 fc
    ┗━ 10 -> net : 20 fc
    -> (20, 20)
    >>> _ = Layer((10,20),1,"bfc")
    (10, 20) seq [1] ['bfc']
    ┗━ (10, 20) -> net : 1 bfc
    -> 1
    >>> _ = Layer(10,[(10,20),1],["fc","bfc"])
    10 seq [(10, 20), 1] ['fc', 'bfc']
    ┗━ 10 loc (10, 20) ('fc', 'fc')
       ┗━ 10 -> net : 10 fc
       ┗━ 10 -> net : 20 fc
       -> (10, 20)
    ┗━ (10, 20) -> net : 1 bfc
    -> 1
    >>> _ = Layer((10,20),[(10,20)],["bfc","bfc"])
    (10, 20) seq [(10, 20), (10, 20)] ['bfc', 'bfc']
    ┗━ (10, 20) loc (10, 20) ('bfc', 'bfc')
       ┗━ (10, 20) -> net : 10 bfc
       ┗━ (10, 20) -> net : 20 bfc
       -> (10, 20)
    ┗━ (10, 20) loc (10, 20) ('bfc', 'bfc')
       ┗━ (10, 20) -> net : 10 bfc
       ┗━ (10, 20) -> net : 20 bfc
       -> (10, 20)
    -> (10, 20)
    >>> _ = Layer(2,[(1,1),2,1],[["fc","act"],"cat",["fc","act"]])
    2 seq [(1, 1), 2, 1] [['fc', 'act'], 'cat', ['fc', 'act']]
    ┗━ 2 loc (1, 1) (['fc', 'act'], ['fc', 'act'])
       ┗━ 2 seq [1, 1] ['fc', 'act']
          ┗━ 2 -> net : 1 fc
          ┗━ 1 -> net : 1 act
          -> 1
       ┗━ 2 seq [1, 1] ['fc', 'act']
          ┗━ 2 -> net : 1 fc
          ┗━ 1 -> net : 1 act
          -> 1
       -> (1, 1)
    ┗━ (1, 1) -> net : 2 cat
    ┗━ 2 seq [1, 1] ['fc', 'act']
       ┗━ 2 -> net : 1 fc
       ┗━ 1 -> net : 1 act
       -> 1
    -> 1
    """
    if isinstance(i,(str,list)):
        net = i
        i = 0

    name = ''
    def mode_check(net,o):
        net_list = net if isinstance(net,(list,tuple))  else [net]
        o_list = o if isinstance(o,(list,tuple)) else [o]
        max_len = max(len(net_list),len(o_list))
        if len(net_list) == 1 and len(o_list) == 1:
            mode = "seq"
            return mode,list(o_list),list(net_list)
        elif isinstance(net_list,(list)) and isinstance(o_list,(list)):
            mode = "seq"
            net_list.extend([net_list[-1]]*(max_len-len(net_list)))
            o_list.extend([o_list[-1]]*(max_len-len(o_list)))
        elif isinstance(net_list,(list)) and isinstance(o_list,(tuple)):
            mode = "loc"
            if len(net_list) == 1:
                net_list = net_list[0]
            net_list = (net_list,)*len(o_list)
        elif isinstance(net_list,(tuple)) and isinstance(o_list,(list)):
            mode = "loc"
            if len(o_list) == 1:
                o_list = o_list[0]
            o_list = (o_list,)*len(net_list)
        else:
            list(net_list).extend([net_list[-1]]*(max_len-len(net_list)))
            list(o_list).extend([o_list[-1]]*(max_len-len(o_list)))
            net_list = tuple(net_list)
            o_list = tuple(o_list)
            mode = "loc"
        return mode,o_list,net_list

    mode,o_list,net_list = mode_check(net,o)

    net_name = "net"
    Nets = Fn(call=mode,name = net_name)
    if print_:
        if gap == 0:
            print(i,mode,o_list,net_list)
        else:
            if len(o_list) > 1 or len(net_list) > 1:
                print("   "*(gap-1)+"┗━ "+str(i),mode,o_list,net_list)
    next_i_list=[]
    for k,(o,net) in enumerate(zip(o_list , net_list)):
        if isinstance(o,(list,tuple)) and isinstance(net,(list,tuple)):
            if len(o) == 1 and len(list) == 1:
                o = o[0]
                net = net[0]
        if not isinstance(o,(list,tuple)) and not isinstance(net,(list,tuple)):
            assert isinstance(net,(str,Callable)),f"{net} is not a string or Callable"
            assert isinstance(o, int),f"{o} is not an int"
            name = net if isinstance(net,str) else net.__name__
            if print_:
                print("   "*(gap)+"┗━ "+str(i),"->", "net :",o,name)
            if isinstance(net,str):
                name = get_args(net)[0]
                func = defdef.get(net) if defdef is not None else mn_get(net)
            try:
                func = func()
            except Exception:
                pass
            finally:
                func = func.func if Fn.is_ddf(func) else func
                if list(inspect.signature(func).parameters.keys()) == ["i","o"]:
                    func = func(i,o)
                else:
                    func = func
                    o = i
            Net = monet(func,i,o,net,defdef)
            Nets.add_module(f"{k}:{name}", Net )

        else:
            Net = Layer(i,o,net,in_dim,defdef,gap+1,print_)
            if isinstance(net,(list,tuple)) and isinstance(o,(list,tuple)):
                Nets.add_module(f"{k}-group",Net)
            else:
                Nets.add_module(f"{k}-cell",Net)
        i = o[-1] if isinstance(o,list) else o  if mode == "seq" else i
        next_i_list = [o] if mode == "seq" else next_i_list+[o]
    if print_:
        next_i_list = next_i_list[0] if len(next_i_list) == 1 else tuple(next_i_list)
        print("   "*gap+"->",next_i_list)
    return Nets

X = Fn()

if __name__ == "__main__":
    import doctest

    doctest.testmod()
