from collections import OrderedDict
from typing import Callable

try:
    import torch.nn as nn
    Module_base = nn.Module
except:
    Module_base = None

is_pcall = False

# 定义一个函数，用于获取函数的名字
def get_name(func):
    # 获取函数的表示形式
    fr=repr(func)
    # 获取函数的类名
    fn=func.__class__.__name__
    # 如果函数的表示形式以函数的类名开头，则返回函数的表示形式
    if fr.startswith(fn):
        return fr
    # 如果函数的类名是'builtin_function_or_method'或者'function'，则返回函数的__name__属性
    elif fn in ['builtin_function_or_method','function']:
        return func.__name__
    # 否则，返回函数的类名
    else:
        return fn

# 定义SEQ函数，用于接收参数ord，args和kwargs，并返回y
def SEQ(ord,*args,**kwargs):
    # 如果args为空，则将x赋值为空列表，否则将args转换为列表赋值给x
    x = [] if len(args) == 0 else [*args]
    # 如果kwargs为空，则将x赋值为空列表，否则将kwargs转换为列表赋值给x
    x+= [] if len(kwargs) == 0 else [kwargs]
    # 将x赋值给y
    y = x = (*x,)
    # 如果ord的长度大于0，则遍历ord中的每一个元素，并将其赋值给y
    if len(ord) > 0:
        for i,func in ord.items():
            y = func(*x)
            # 如果mstr不等于')'且is_pcall为真，则打印出i，mstr，x和y
            if (mstr:=repr(func).split('\n')[-1]) != ')' and is_pcall:
                print(f"  {i} : "+mstr+f"\n  $ {x} >>\n  ==", y )
            # 将y赋值给x
            x = (y,)
        # 如果y的长度大于0，则返回y
        if len((y,))>0:
            return y
        
# 定义函数LIC，参数ord，*args，**kwargs，返回值all_y
# ord：字典，存储函数和参数
# args：可变参数，参数列表
# kwargs：关键字参数，字典
# x：列表，存储参数列表
# y：列表，存储函数的返回值
# all_y：列表，存储所有函数的返回值
# 如果ord不为空，则遍历ord，将函数的返回值赋值给y，如果mstr不等于')'且is_pcall为真，则打印函数和参数以及函数的返回值
# 最后返回all_y，如果all_y的长度大于0，则返回all_y，否则返回None
def LIC(ord,*args,**kwargs):
    '''
    ord：字典，存储函数和参数
    args：可变参数，参数列表
    kwargs：关键字参数，字典
    x：列表，存储参数列表
    y：列表，存储函数的返回值
    all_y：列表，存储所有函数的返回值
    如果ord不为空，则遍历ord，将函数的返回值赋值给y，如果mstr不等于')'且is_pcall为真，则打印函数和参数以及函数的返回值
    最后返回all_y，如果all_y的长度大于0，则返回all_y，否则返回None
    '''
    x = [] if len(args) == 0 else [*args]
    x+= [] if len(kwargs) == 0 else [kwargs]
    y = x = (*x,)
    all_y = []
    if len(ord) > 0:
        for i,func in ord.items():
            y = func(*x)
            if (mstr:=repr(func).split('\n')[-1]) != ')' and is_pcall:
                print(f"  {i} : "+mstr+f"\n  $ {x} >>\n  ==", y )
            all_y += [y]
        if len((all_y,))>0 :
            return all_y




            
class FuncModel(Module_base): # type: ignore
            
    # 定义一个函数，参数args为一个列表，call为一个函数
    def __init__(self,args=[],call:str | Callable ='SEQ'):
        # 调用父类的初始化函数
        super().__init__()
        # 如果args不是一个可迭代对象，则将其转换为列表
        args = [args] if not hasattr(args,'__iter__') else args
        # 初始化一个有序字典
        self._modules=OrderedDict()
        # 遍历args中的每一个元素
        for i,arg in enumerate(args):
            # 如果arg是一个Module或者dup对象
            if issubclass(arg.__class__,(FuncModel,dup)):
                # 将arg中的第一个元素添加到有序字典中，如果arg的元素个数不为0，且arg的元素个数为1，则将arg中的第一个元素添加到有序字典中
                self._modules[str(i)] = [*arg][0] if repr(arg).count('\n') !=0 and len(arg)==1 else arg
            # 如果arg不是一个Module或者dup对象
            else:
                # 将arg复制一份添加到有序字典中
                self._modules[str(i)] = self.dup(arg)
        # 将call赋值给self.call
        self.call = eval(call) if isinstance(call,str) else call
        # 将self.p赋值为self.pcall
        self.p = self.pcall

    def __call__(self,*args,**kwargs):
        '''
        调用call函数，并打印出函数的参数和输出结果
        '''
        global is_pcall
        if is_pcall:
            print('@',self.__repr__().split('\n')[0],'>>\n   ', *args, **kwargs)
        output = self.call(self._modules,*args,**kwargs)
        if is_pcall:
            print(f')>>', output)
        return output
    
    def pcall(self,*args,**kwargs):
        # 定义一个全局变量is_pcall，用于判断是否调用pcall函数
        global is_pcall
        # 将is_pcall设置为True
        is_pcall =True
        # 调用父类的__call__函数，并将返回值赋值给output
        output = self.__call__(*args,**kwargs)
        # 将is_pcall设置为False
        is_pcall =False
        # 返回output
        return output
    
    def __iter__(self):
        return iter(self._modules.values())

    def __getitem__(self,i):
        return self._modules.__getitem__(i)
    
    def  __repr__(self):
        # 获取父类的__repr__方法
        main_str = super().__repr__()
        # 如果父类的__repr__方法以当前类的名称开头，则返回当前类的名称和父类的__repr__方法
        if main_str.startswith(self.__class__.__name__):
            return f"{get_name(self.call)}>"+main_str
        # 否则，返回当前类的名称和当前类的__repr__方法
        else:
            lines = [self.__class__.__name__+'(']
            # 遍历当前类的_modules属性，获取每个模块的__repr__方法
            for key,func in self._modules.items():
                _rerp_str = []
                # 遍历每个模块的__repr__方法，添加缩进
                for i,r in enumerate(repr(func).split('\n')):
                    _rerp_str+=["  "+r] if i>0 else [r]
                # 将每个模块的__repr__方法添加到lines列表中
                lines+=[f'  ({key}): ' + "\n".join(_rerp_str)]
            # 返回当前类的名称、当前类的__repr__方法和每个模块的__repr__方法
            main_str =f"{get_name(self.call)}>"+'\n'.join(lines)
            return main_str+'\n)' if len(lines)>1 else main_str+')'

    def __len__(self):
        return len(self._modules)
    
    def __add__(self,other):
        '''
        如果other是int类型，则返回self+[Module()]*other，如果other是Module类型，则返回self+[other]，如果other是list类型，则返回self+[Module()]*len(other)，如果other是Module类型，且other.call是SEQ类型，则返回self+[other]，如果other是Module类型，且other.call是LIC类型，则返回self+[other]
        '''
        if isinstance(other,int):
            return self+[FuncModel()]*other if other !=1 else self+[FuncModel()]
        nn_list = [*self] if self.call == LIC else [self]
        nn_list += [*other] if (self.is_dup(other) and other.call == LIC) or ((self.is_dup(other) and other.call != SEQ) and hasattr(other,'__len__')) else [other]
        return FuncModel(nn_list,call=LIC)

    def __mul__(self,other):
        '''
        如果other是int，则返回Module，否则返回Module的列表
        '''
        if isinstance(other,int):
            return FuncModel([self]*other) if other !=1 else self
        else:
            nn_list = [*self] if self.call == SEQ else [self]
            nn_list += [*other] if (self.is_dup(other) and other.call == SEQ) or ((self.is_dup(other) and other.call != LIC) and hasattr(other,'__len__')) else [other]
        return FuncModel(nn_list,call=SEQ)
    
    def __pow__(self,other):
        '''
        如果other是int类型，则返回Module的多个实例，否则返回Module的实例乘以other
        '''
        import copy
        if isinstance(other,int):
            return FuncModel([copy.deepcopy(self) for i in range(other)],call=SEQ) if other !=1 else copy.deepcopy(self)
        else:
            return copy.deepcopy(self)*other
        
    def __and__(self,other):
        '''
        这段代码是一个类的方法，用于重载位运算符"&"。它接受一个参数"other"，根据"other"的类型不同，返回不同的结果。

        代码的步骤如下：
        1. 导入copy模块。
        2. 判断"other"是否是整数类型。
        3. 如果是整数类型，则创建一个新的Module对象的列表，列表的长度为"other"的值。每个新的Module对象都是self的深拷贝。如果"other"不等于1，则返回这个列表加1，否则返回self的深拷贝加1。
        4. 如果"other"不是整数类型，则返回self的深拷贝乘以"other"再加上"other"。'''
        import copy
        if isinstance(other,int):
            return FuncModel([copy.deepcopy(self) for i in range(other)],call=LIC)+1 if other !=1 else copy.deepcopy(self)+1
        else:
            return copy.deepcopy(self)*other+other
    
    def forward(self,*args,**kwargs):
        return self.__call__(*args,**kwargs)
    
    @staticmethod
    def dup(func):
        return dup(func)
    
    @staticmethod
    def is_dup(func):
        return issubclass(func.__class__,(FuncModel,dup))   # type: ignore
    
class dup(FuncModel):
    '''
    这是一个名为dup的类，继承自Module类。它包含一个构造函数__init__，一个调用函数__call__，一个字符串表示函数__repr__。

    类的构造函数将传入的func参数存储在_modules字典中，并将其赋值给func变量。

    __call__函数允许将类的实例作为函数进行调用，它将传入的参数传递给func函数并返回结果。

    __repr__函数返回一个字符串，其中包含类名、_modules字典中存储的函数名称以及类实例的唯一标识符。

    '''
    def __init__(self,func):
        super().__init__()
        self._modules = OrderedDict([('0',func)])
        self.func = func
        
    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)
        
    def __repr__(self):
        main_str=self.__class__.__name__
        return f'@{main_str}:'+ str(repr(self._modules['0']) + f' *id:{id(self)}')
           


    
# 基础类
Fn =  FuncModel

# 形式输入
X =  FuncModel()

# 修饰器
# @Fn
# @dup
# @Fn.dup