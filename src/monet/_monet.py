from typing import Callable, OrderedDict
from monet.flowfunc import FuncModel as Fn
from monet.flowfunc import ddf
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
    >>> mn_get("fc")(10,1)
    Linear(in_features=10, out_features=1, bias=True)
    >>> mn_get("fc_0")(10,1)
    Linear(in_features=10, out_features=1, bias=False)
    >>> mn_get("bfc")((10,20),1)
    Bilinear(in1_features=10, in2_features=20, out_features=1, bias=True)
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


class monet(Fn):
    def __init__(self,module,i=-1,o=0,net='',get=mn_get,in_dim=-1):
        super(monet,self).__init__()
        self.i = i
        self.auto_i = (i == 0)
        self.o = o
        self.net = net
        self.mn_get = mn_get
        self.in_dim = 1 if net.startswith("cv") and in_dim==-1 else in_dim

        self._modules = OrderedDict([('0',module)])
        self.Net = module
        self.func = module

    def forward(self,x,*args,**kwargs):
        if self.auto_i is True and self.i != x.shape[self.in_dim]:
            self.i = x.shape[self.in_dim]
            print("[Warning] Input dim is not match, set to {}".format(self.i))
            self.o,Nets = Layer(self.i,self.o,self.net,self.in_dim,self.get)
            self.Net = Nets.Net
            if hasattr(x,"device") and hasattr(self.Net,"to"):
                self.Net.to(x.device)
        return super().forward(x,*args,**kwargs)

    def __repr__(self):
        main_str=self.__class__.__name__
        return f'@{main_str}:'+ str(repr(self.Net) + f' *id:{id(self)}')


def Layer(i: int | str | list=0,
          o:int | list[int]=1,
          net:str | list[str,Callable] ="fc_1",
          in_dim=-1,
          get = mn_get,gap=0,
          print_=True):
    """
    >>> o, Net = Layer("fc")
    0 seq [1] ['fc']
    ┗━ 0 -> net : 1 fc
    -> 1
    >>> o, Net = Layer(10,1,"fc")
    10 seq [1] ['fc']
    ┗━ 10 -> net : 1 fc
    -> 1
    >>> o, Net = Layer(10,[10,10],"fc")
    10 seq [10, 10] ['fc', 'fc']
    ┗━ 10 -> net : 10 fc
    ┗━ 10 -> net : 10 fc
    -> 10
    >>> o, Net = Layer(10,[10],["fc","act"])
    10 seq [10, 10] ['fc', 'act']
    ┗━ 10 -> net : 10 fc
    ┗━ 10 -> net : 10 act
    -> 10
    >>> o, Net = Layer(10,(10,20),"fc")
    10 lic [10, 20] ['fc', 'fc']
    ┗━ 10 -> net : 10 fc
    ┗━ 10 -> net : 20 fc
    -> (10, 20)
    >>> o, Net = Layer(10,10,("fc","fc"))
    10 lic [10, 10] ['fc', 'fc']
    ┗━ 10 -> net : 10 fc
    ┗━ 10 -> net : 10 fc
    -> (10, 10)
    >>> o, Net = Layer((10,20),1,"bfc")
    (10, 20) seq [1] ['bfc']
    ┗━ (10, 20) -> net : 1 bfc
    -> 1
    >>> o, Net = Layer(10,[(10,20),1],["fc","bfc"])
    10 seq [(10, 20), 1] ['fc', 'bfc']
    ┗━ 10 -> lic : (10, 20) fc
       ┗━ 10 -> net : 10 fc
       ┗━ 10 -> net : 20 fc
       -> (10, 20)
    ┗━ (10, 20) -> net : 1 bfc
    -> 1
    >>> o, Net = Layer((10,20),(10,20),["bfc","bfc"])
    (10, 20) seq [(10, 20), (10, 20)] ['bfc', 'bfc']
    ┗━ (10, 20) -> lic : (10, 20) bfc
       ┗━ (10, 20) -> net : 10 bfc
       ┗━ (10, 20) -> net : 20 bfc
       -> (10, 20)
    ┗━ (10, 20) -> lic : (10, 20) bfc
       ┗━ (10, 20) -> net : 10 bfc
       ┗━ (10, 20) -> net : 20 bfc
       -> (10, 20)
    -> (10, 20)
    """
    if isinstance(i,(str,list)) :
        net = i
        i = 0

    if isinstance(o,(tuple))  and not isinstance(net,(list)):
        mode = "lic"
    elif isinstance(net,(tuple)) and not isinstance(o,(list)):
        mode = "lic"
    else:
        mode = "seq"
    Nets = Fn(call=mode)

    net_list = net if isinstance(net,list) else list(net) if isinstance(net,tuple) and not isinstance(o,(list)) else [net]
    o_list = o if isinstance(o,list) else list(o) if isinstance(o,tuple) and not isinstance(net,(list)) else [o]

    max_len = max(len(net_list),len(o_list))
    net_list.extend([net_list[-1]]*(max_len-len(net_list)))
    o_list.extend([o_list[-1]]*(max_len-len(o_list)))

    # o in tuple means Ndim output
    # net in tuple means lic output
    if gap == 0 and print_:
        print(i,mode,o_list,net_list)
    next_i_list=[]
    for k,(o,net) in enumerate(zip(o_list , net_list)): # type: ignore
        if isinstance(o,(tuple))  and not isinstance(net,(list)):
            cmode = "lic"
        elif isinstance(net,(tuple)) and not isinstance(o,(list)):
            cmode = "lic"
        else:
            cmode = "seq"
        if isinstance(o,(list,tuple)) or isinstance(net,(list,tuple)):
            if print_:
                print("   "*(gap)+"┗━ "+str(i),"->", cmode,":", o,net)
        else:
            if print_:
                print("   "*(gap)+"┗━ "+str(i),"->", "net :", o, net)

        if isinstance(o,(list,tuple)) and isinstance(net,(list,tuple)):
            next_i, module = Layer(i,o,net,in_dim,get,gap+1,print_)
            Nets.add_module(f"{k}:mix",module)
            i = next_i if mode == "seq" else i
            next_i_list = next_i_list+next_i if mode == "lic" else next_i
        elif isinstance(net,(list,tuple)):
            next_i, module = Layer(i,o,net,in_dim,get,gap+1,print_)
            Nets.add_module(f"cell-{k}",module)
            i = next_i if mode == "seq" else i
            next_i_list = next_i_list+next_i if mode == "lic" else next_i
        elif isinstance(o,(list,tuple)):
            next_i, module = Layer(i,o,net,in_dim,get,gap+1,print_)
            Nets.add_module(f"{k}:{net} x {len(module)}",module)
            i = next_i if mode == "seq" else i
            next_i_list = next_i_list+next_i if mode == "lic" else next_i
        else:
            assert isinstance(net,(str,Callable)),f"{net} is not a string or Callable"
            name = ''
            if isinstance(net,str):
                if net == '':
                    Net = monet()
                else:
                    import inspect
                    func = get(net)
                    if list(inspect.signature(func.func).parameters.keys()) == ["i","o"]:
                        Net = func(i,o)
                    else:
                        Net = func
                    name = get_args(net)[0]
            else:
                try:
                    Net = net(i,o)
                except Exception:
                    Net = net
                name = net.__name__
            Net = monet(Net,i,o,net,get,in_dim) # type: ignore
            if i == 0:
                Nets.in_dim = Net.in_dim
            Nets.add_module(f"{k}:{name}", Net )
            if mode == 'seq':
                i = o
                next_i_list = [o]
            else:
                i = i
                next_i_list += [o]
    if print_:
        print("   "*gap+"->",tuple(next_i_list) if len(next_i_list) > 1 else next_i_list[0])
    return tuple(next_i_list), Nets

X = Fn()

if __name__ == "__main__":
    import doctest

    doctest.testmod()
