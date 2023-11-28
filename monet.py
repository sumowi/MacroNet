import torch
import torch.nn as nn

# 默认预定义的模块
default_dict={
    "fc_True": lambda i,o,bias: 
        nn.Linear(in_features=i,out_features=o,bias=bias),
    "bfc_True": lambda i,o,bias: 
        nn.Bilinear(in1_features=i[0],in2_features=i[1],out_features=o,bias=bias),
    "flat_1_-1": lambda i,o,start_dim,end_dim: 
        nn.Flatten(start_dim=start_dim,end_dim=end_dim),
        
    # 卷积核带维度数，默认2维
    "cov2_3_1_0_True": lambda i,o,dim,kernel_size,stride,padding,bias:
        eval(f"nn.Conv{dim}d")(in_channels=i,out_channels=o,kernel_size=kernel_size,
                               stride=stride,padding=padding,bias=bool(bias)), # 卷积
    "covT2_3_1_0_True": lambda i,o,dim,kernel_size,stride,padding,bias:
        eval(f"nn.ConvTranspose{dim}d")(in_channels=i,out_channels=o,kernel_size=kernel_size,
                                        stride=stride,padding=padding,bias=bias), # 反卷积
    # 最大池化带维度数，默认2维 
    "mp2_2_0": lambda i,o,dim,kernel_size,padding:
        eval(f"nn.MaxPool{dim}d")(kernel_size=kernel_size,padding=padding), # 最大池化
    "amp2_2_0": lambda i,o,dim,kernel_size,padding:
        eval(f"nn.AdaptiveMaxPool{dim}d")(kernel_size=kernel_size,padding=padding), # 自适应最大池化
        
    # 平均池化带维度数，默认2维 
    "ap2_2": lambda i,o,dim,padding:
        eval(f"nn.AvgPool{dim}d")(padding=padding), # 平均池化
    "aap2_2": lambda i,o,dim,padding:
        eval(f"nn.AdaptiveAvgPool{dim}d")(padding=padding), # 自适应平均池化
        
    # 归一化带维度,默认1维
    "bn1_0": lambda i,o,dim,num_features:
        eval(f"nn.BatchNorm{dim}d")(num_features=i if num_features==0 else num_features),
    "in1_0": lambda i,o,dim,num_features:
        eval(f"nn.InstanceNorm{dim}d")(num_features=i if num_features==0 else num_features),
        
    # 其他归一化
    "gn_2": lambda i,o,num_groups:
        nn.GroupNorm(num_groups=num_groups,num_channels=i),
    "ln": lambda i,o:
        nn.LayerNorm(normalized_shape=i),
    "lrn": lambda i,o:
        nn.LocalResponseNorm(size=i),
        
    # 随机神经元丢弃
    "dp_0.5_False": lambda i,o,p,inplace:
        nn.Dropout(p=p,inplace=bool(inplace)),
    "dp1_1_False": lambda i,o,dim,p,inplace:
        eval(f"nn.Dropout{dim}d")(p=p,inplace=bool(inplace)),
    
    # AlphaDropOut
    "aldp_0.5_False": lambda i,o,p,inplace:
        nn.AlphaDropout(p=p,inplace=bool(inplace)),
    "fadp_0.5_False": lambda i,o,p,inplace:
        nn.FeatureAlphaDropout(p=p,inplace=bool(inplace)),
    
    # 激活函数，.传递字符串参数
    "act.PReLU": lambda i,o,act_func: 
        eval(f"nn.{act_func}")(),
    # 任意nn模块，()传递所有参数
    "nn.Linear_(10,1)": lambda i,o,func,args: 
        eval(f"nn.{func}")(*args),
    
}

# 额外的模块，会优先于default_dict被检查
mn_dict={
    
    }

def get_args(net="fc_1"):
    # _分隔值形参数，.分割字符串形式参数
    # .在_前面,只能传递一个参数,会在参数名称里
    args=net.split("_")
    args_str = args[0].split(".")
    name = args_str[0]
    # 先添加字符串形式参数
    args_str = args_str[1:]
    # 从name的最后一个字符提取维度值
    args_opt = [eval(name[-1])] if name[-1] in "1234567890" else []
    name = name[:-1] if name[-1] in "1234567890" else name
    # 添加值形式从参数,''返回空值
    args_opt += [eval(i) if i !='' else [] for i in args[1:] ]
    return name,args_str,args_opt

def eval_mn(i=10,o=1,name='fc',args_str=[],args_opt=[],mn_dicts=default_dict):
    for monet in mn_dicts.keys():
        if monet.startswith(name):   # type: ignore
            mo_name,mo_args_str,mo_args_opt=get_args(monet)
            args = args_str if len(args_str)==1 else mo_args_str
            for n,s in enumerate((mo_args_opt)):
                if args_opt[n:n+1] == []:
                    # 如果是空值，使用默认值
                    args += [s]
                else:
                    # 否则使用设定值
                    args += args_opt[n:n+1]
            return mn_dicts[monet](i,o,*args)
    return 0

class MoNet(nn.Module):
    def __init__(self,Module=[],mode=''):
        super(MoNet,self).__init__()
        self.mode = mode
        match mode:
            case "" :
                self.Net = Module
            case "*":
                if len(Module)==1:
                    self.Net = Module[0]
                else:
                    self.Net = nn.Sequential(*Module)
            case "+":
                self.Net = nn.ModuleList(Module)
    
    def __iter__(self):
        return iter(self.Net)
    
    def __mul__(self, other):
        res = [self,other]
        return MoNet(Module=res,mode='*')
    
    def __add__(self, other):
        res = [self,other]
        return MoNet(Module=res,mode='+')
    
    # TODO: 写得有问题
    # def __repr__(self):
    #     if self.mode == "*":
    #         return f'MoNet({self.__repr__})'
    #     elif self.mode == "+":
    #         return f'MoNet(Cat>{self.__repr__})'
    
    def forward(self, x):
        if self.mode == "":
            return self.Net(x)
        elif self.mode == "*":
            return self.Net(x)
        elif self.mode == "+":
            x1 = self.Net[0](x)
            x2 = self.Net[1](x)
            return torch.cat([x1,x2],dim=0)

def layer(i=10,o=1,net="fc_1",mn_dict=mn_dict):
    # 获取参数
    args=get_args(net)
    Net = eval_mn(i,o,*args,mn_dict)
    if Net==0:
        Net = eval_mn(i,o,*args,default_dict)
        assert Net!=0,f"No such layer {net}"
    return MoNet(Net)

class Layer(MoNet):
    def __init__(self,i=10,o=1,net="fc_1",mn_dict=mn_dict,in_dim=1):
        super(Layer,self).__init__()
        self.net = net
        self.i = i
        self.o = o
        self.mn_dict = mn_dict
        self.in_dim = in_dim
        self.Net=layer(i,o,net,mn_dict).Net
        
    def forward(self,x):
        if x.shape[self.in_dim] != self.i:
            self.i = x.shape[self.in_dim]
            self.Net = layer(self.i,self.o,self.net,self.mn_dict).Net
        return self.Net(x)

def seqLayer(i=10,o_list=[64,64],net="fc_1",mn_dict=mn_dict):
    # 获取参数
    o_list = [o_list] if type(o_list)==int else o_list
    Net = nn.Sequential()
    for n,o in enumerate(o_list):
        Net.add_module(f"{net}_{n}",Layer(i,o,net,mn_dict).Net)
    return MoNet(Net)

class SeqLayer(MoNet):
    def __init__(self,i=10,o_list=[64,64],net="fc_1",mn_dict=mn_dict,in_dim=1):
        super(SeqLayer,self).__init__()
        self.net = net
        self.i = i
        self.o = o_list
        self.mn_dict = mn_dict
        self.in_dim = in_dim
        self.Net=seqLayer(i,o_list,net,mn_dict).Net
        
    def forward(self,x):
        if x.shape[self.in_dim] != self.i:
            self.i = x.shape[self.in_dim]
            self.Net = layer(self.i,self.o,self.net,self.mn_dict).Net
        return self.Net(x)

def cell(i=10,o=1,net_list=["fc","bn","act","dp"],mn_dict=mn_dict):
    # 获取参数
    Net = nn.Sequential()
    for n,net in enumerate(net_list):
        name,_,_=get_args(net)
        Net.add_module(f"{n}:{name}",Layer(i,o,net,mn_dict).Net)
        i=o
    return MoNet(Net)

class Cell(MoNet):
    def __init__(self,i=10,o=1,net_list=["fc","bn","act","dp"],mn_dict=mn_dict, in_dim=1):
        super(Cell,self).__init__()
        self.net = net_list
        self.i = i
        self.o = o
        self.mn_dict = mn_dict
        self.in_dim = in_dim
        self.Net=cell(i,o,net_list,mn_dict).Net
        
    def forward(self,x):
        if x.shape[self.in_dim] != self.i:
            self.i = x.shape[self.in_dim]
            self.Net[0] = layer(self.i,self.o,self.net[0],self.mn_dict).Net
        return self.Net(x)

def seqCell(i=10,o_list=[10,1],net_list=["fc","bn","act","dp"],name='cell',mn_dict=mn_dict):
    # 获取参数
    o_list = [o_list] if type(o_list)==int else o_list
    net_list = [net_list] if type(net_list)==str else net_list
    
    Net = nn.Sequential()
    for n,o in enumerate(o_list):
        Net.add_module(f"{name}-{n}",Cell(i,o,net_list,mn_dict).Net)
        i=o
    return MoNet(Net)

class SeqCell(MoNet):
    def __init__(self,i=10,o_list=[10,1],net_list=["fc","bn","act","dp"],name='cell',mn_dict=mn_dict, in_dim=1):
        super(SeqCell,self).__init__()
        self.net = net_list
        self.i = i
        self.o = o_list
        self.mn_dict = mn_dict
        self.in_dim = 1
        self.Net=seqCell(i,o_list,net_list,name,mn_dict).Net
        
    def forward(self,x):
        if x.shape[self.in_dim] != self.i:
            self.i = x.shape[self.in_dim]
            self.Net[0] = layer(self.i,self.o[0],self.net[0],self.mn_dict).Net
        return self.Net(x)

def mix(i=10,o_lists=[10,[32,32],1],net_lists=['dp_0.2',["fc",'bn','act','dp_0.5'],"fc"],name_list=['input','hiddens','out'],mn_dict=mn_dict):
    o_lists = [o_lists] if type(o_lists)==int else o_lists
    net_lists = [net_lists] if type(net_lists)==str else net_lists
    name_list = [name_list] if type(name_list)==str else name_list
    
    cell_num=max(len(net_lists),len(o_lists))
    if len(net_lists)==1:
        net_lists=[net_lists[0]]*cell_num
    if len(o_lists)==1:
        o_lists=[o_lists[0]]*cell_num
    if len(name_list)==1:
        name_list=[name_list[0]]*cell_num
    
    Net=nn.Sequential()
    for n,(net_list,o) in enumerate(zip(net_lists, o_lists)):
        o = [o] if type(o) == int else o
        net_list = [net_list] if type(net_list) == str else net_list
        Net.add_module(f'{n}:{name_list[n]}',SeqCell(i,o,net_list,'cell',mn_dict).Net) # type: ignore
        i = o[-1] # type: ignore
    return MoNet(Net)

class Mix(MoNet):
    def __init__(self,i=10,o_lists=[10,[32,32],1],net_lists=['dp_0.2',["fc",'bn','act','dp_0.5'],"fc"],name_list=['input','hiddens','out'],mn_dict=mn_dict, in_dim = 1):
        super(Mix,self).__init__()
        self.net_lists = net_lists
        self.i = i,
        self.o = o_lists,
        self.mn_dict = mn_dict
        self.in_dim = in_dim
        self.Net=mix(i,o_lists,net_lists,name_list,mn_dict).Net
        
    def forward(self,x):
        if x.shape[self.in_dim] != self.i:
            self.i = x.shape[self.in_dim]
            self.Net[0] = layer(self.i,self.o[0],self.net[0],self.mn_dict).Net
        return self.Net(x)
    
