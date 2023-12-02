from re import X
import re
from typing import Callable
import torch
import torch.nn as nn
from .flowfunc import FuncModel
# 额外的模块，会优先于default_dict被检查
mn_dict={
    
    }

# 默认预定义的模块
default_dict={
    "fc_True": lambda i,o,bias: 
        nn.Linear(in_features=i,out_features=o,bias=bias),
    "bfc_True": lambda i,o,bias: 
        nn.Bilinear(in1_features=i[0],in2_features=i[1],out_features=o,bias=bias),
    "fl_1_-1": lambda i,o,start_dim,end_dim: 
        nn.Flatten(start_dim=start_dim,end_dim=end_dim),
        
    # 卷积核带维度数，默认2维
    "cv2_3_1_0_True": lambda i,o,dim,kernel_size,stride,padding,bias:
        eval(f"nn.Conv{dim}d")(in_channels=i,out_channels=o,kernel_size=kernel_size,
                               stride=stride,padding=padding,bias=bool(bias)), # 卷积
    "cvT2_3_1_0_True": lambda i,o,dim,kernel_size,stride,padding,bias:
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
    "adp_0.5_False": lambda i,o,p,inplace:
        nn.AlphaDropout(p=p,inplace=bool(inplace)),
    "fadp_0.5_False": lambda i,o,p,inplace:
        nn.FeatureAlphaDropout(p=p,inplace=bool(inplace)),
    
    # 激活函数，.传递字符串参数
    "act.GELU": lambda i,o,act_func:
        eval(f"nn.{act_func}")(),
    # 任意nn模块，()传递所有参数
    "nn.Linear_(10,1)": lambda i,o,func,args: 
        eval(f"nn.{func}")(*args),
    
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

def eval_mn(i=0,o=1,name='fc',args_str=[],args_opt=[],mn_dicts=default_dict):
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


# 通过字符串形式构建函数
class FLOW(FuncModel):
    def __init__(self,module=[],i=0,o=0,net='',mn_dict={},in_dim=-1):
        super(FLOW,self).__init__()
        self.i = i
        self.auto_i = (i == 0)
        self.o = o
        self.net =net
        self.in_dim = 1 if net.startswith("cv") and in_dim==-1 else in_dim
        self.mn_dict = mn_dict
        if module != []:
            self.Net=self.dup(module)
    
    def forward(self,x,*args,**kwargs):
        if self.auto_i == True and self.i != x.shape[self.in_dim]:
            self.i = x.shape[self.in_dim]
            self.Net = layer(self.i,self.o,self.net,self.mn_dict).Net
            self.Net.to(x.device)
        return self.Net(x,*args,**kwargs)


def layer(i: int | str | list=0,
          o:int | list=1,
          net:str | list ="fc_1",
          mn_dict=mn_dict,
          in_dim=-1):
    if isinstance(i,(str,list)) : 
        net = i ; i = 0 
    
    net_list = [net] if isinstance(net,str) else net
    o_list = [o] if isinstance(o,(int,tuple)) else o
    
    max_len = max(len(net_list),len(o_list)) # type: ignore
    
    net_list.extend([net_list[-1]]*(max_len-len(net_list))) # type: ignore
    o_list.extend([o_list[-1]]*(max_len-len(o_list))) # type: ignore
    
    print(o_list,net_list)
    Nets = FLOW()
    for k,(o,net) in enumerate(zip(o_list , net_list)): # type: ignore
        if isinstance(o,(list,tuple)) and isinstance(net,(list,tuple)):
            Nets.add_module(f"{k}:mix",layer(i,o,[net],mn_dict,in_dim))
            i = o[-1]
        elif isinstance(net,(list,tuple)):
            Nets.add_module(f"cell-{k}",layer(i,o,net,mn_dict,in_dim))
            i = o
        elif isinstance(o,(list,tuple)):
            Net = layer(i,o,[net],mn_dict,in_dim)
            Nets.add_module(f"{k}:{net} x {len(Net)}",nn.Sequential(*Net))
            i = o[-1]
        else:
            assert isinstance(net,(str,Callable)),f"{net} is not a string or Callable"
            name = ''
            if isinstance(net,str):
                # 获取参数
                args=get_args(net)
                Net = eval_mn(i,o,*args,mn_dict)
                if Net==0:
                    Net = eval_mn(i,o,*args,default_dict)
                    assert Net!=0,f"No such layer {net}"
                name = args[0]
            else:
                try:
                    Net = net(i,o)
                except:
                    Net = net
                name = net.__name__ 
            Net = FLOW(Net,i,o,net,mn_dict,in_dim)
            if i == 0: Nets.in_dim = Net.in_dim
            Nets.add_module(f"{k}:{name}", Net.Net )
            i = o
    return Nets

class Mix(FLOW):
    def __init__(self,i=0,o_lists=[10,[32,32],1],net_lists=['dp_0.5',["fc",'bn','act','dp_0.5'],"fc"],mn_dict=mn_dict,in_dim=-1):
        super(Mix,self).__init__()
        self.i = i,
        self.auto_i = (i == 0)
        self.o = o_lists if isinstance(o_lists,int) else o_lists[-1],
        self.o_list =  [o_lists] if isinstance(o_lists,int) else o_lists
        self.net = [o_lists] if isinstance(net_lists,str) else net_lists
        self.mn_dict = mn_dict
        self.Net=layer(i,o_lists,net_lists,mn_dict,in_dim)
        self.in_dim = self.Net.in_dim
        
    def forward(self,x,*args,**kwargs):
        return self.Net(x,*args,**kwargs)
