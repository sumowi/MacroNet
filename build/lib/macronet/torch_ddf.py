"""
>>> from macronet.base import MoNetInitial,dict_slice
>>> from macronet.torch_ddf import torch_dict
>>> m = MoNetInitial(dict_slice(torch_dict,0,28))
>>> m("fc_False")(10,1).name
'@ddf:Linear(in_features=10, out_features=1, bias=False)'
>>> m("bfc_0")((10,10),1).name
'@ddf:Bilinear(in1_features=10, in2_features=10, out_features=1, bias=False)'
>>> m.get("act").func
act ('act.PReLU_()', {'activation': 'PReLU', 'args': ()})
PReLU(num_parameters=1)
>>> m.get("relu").func
relu ('relu_b', {'b': False})
ReLU()
>>> m.find("prelu0.1")
('prelu_p', {'_p': 0.1})
>>> print(len(m.funcspace))
29
"""

import torch.nn as nn
import torch

torch_dict = {
    # Fully Connected Layer
    "fc_True": lambda bias: lambda i, o: nn.Linear(
        in_features=i, out_features=o, bias=bias
    ),
    # Bilinear Fully Connected Layer
    "bfc_True": lambda bias: lambda i, o: nn.Bilinear(
        in1_features=i[0], in2_features=i[1], out_features=o, bias=bias
    ),
    # Activation function,. passing string parameters
    "act.PReLU_()": lambda activation, args:
        eval(f"nn.{activation}")(*args),
    # Common activation functions
    "relu_b": lambda b=False: nn.ReLU(inplace=b),
    "prelu_p": lambda _p=0.25: nn.PReLU(init=_p),
    "gelu_s": lambda s="none": nn.GELU(str=s),
    "softmax_d": lambda d=None: nn.Softmax(dim=d),
    "tanh": nn.Tanh(),
    "sigmoid": nn.Sigmoid(),
    "cross_()": lambda args: nn.CrossEntropyLoss(*args),
    # ======================================================
    # DropOut, DropOutNdim, AlphaDropout, FeatureAlphaDropout
    # bouth have p and innplace option
    "dp_0.5_False": lambda pv, inplace: nn.Dropout(
        p=pv, inplace=bool(inplace)
    ),
    "dp1_1_False": lambda Ndim, pv, inplace: eval(f"nn.Dropout{Ndim}d")(
        p=pv, inplace=bool(inplace)
    ),
    # AlphaDropOut
    "adp_0.5_False": lambda p, inplace: nn.AlphaDropout(
        p=p, inplace=bool(inplace)
    ),
    "fadp_0.5_False": lambda p, inplace: nn.FeatureAlphaDropout(
        p=p, inplace=bool(inplace)
    ),
    # Normalized, InstanceNorm with dimensions, default to 1 dimension
    "bn1_0": lambda dim, num_features: lambda i, o: eval(f"nn.BatchNorm{dim}d")(
        num_features=i if num_features == 0 else num_features
    ),
    "in1_0": lambda dim, num_features: lambda i, o: eval(f"nn.InstanceNorm{dim}d")(
        num_features=i if num_features == 0 else num_features
    ),
    # GroupNorm by num_groups
    "gn_2": lambda num_groups: lambda i, o: nn.GroupNorm(
        num_groups=num_groups, num_channels=i
    ),
    "ln_[]": lambda shape: lambda i, o: nn.LayerNorm(
        normalized_shape=i if shape == [] else shape
    ),
    "lrn": lambda i, o: nn.LocalResponseNorm(size=i),
    "syn": lambda i, o: nn.SyncBatchNorm(i),
    # ======================================================
    # N-dimensional convolution
    "cv2_3_1_0": lambda Ndim, kernel_size, stride, padding: lambda i, o: eval(f"nn.Conv{Ndim}d")(
        in_channels=i, out_channels=o, kernel_size=kernel_size, stride=stride, padding=padding
    ),
    # Transposed convolution
    "cvT2_3_1_0": lambda Ndim, kernel_size, stride, padding: lambda i, o: eval(
        f"nn.ConvTranspose{Ndim}d"
    )(in_channels=i, out_channels=o, kernel_size=kernel_size, stride=stride, padding=padding),  # 反卷积
    # N-dimensional Maximum pooling
    "mp2_3_1_0": lambda dim, kernel_size, stride, padding: eval(f"nn.MaxPool{dim}d")(
        kernel_size=kernel_size, stride=stride, padding=padding
    ),
    # Adaptive N-dimensional Maximum pooling
    "amp2_3_1_0": lambda dim, kernel_size, stride, padding: eval(
        f"nn.AdaptiveMaxPool{dim}d"
    )(kernel_size=kernel_size, stride=stride, padding=padding),  # 自适应最大池化
    # N-dimensional Avarage pooling
    "ap2_3_1_0": lambda dim, kernel_size, stride, padding: eval(f"nn.AvgPool{dim}d")(
        kernel_size=kernel_size, stride=stride, padding=padding
    ),
    # Adaptive N-dimensional Avarage pooling
    "aap2_(6,6)": lambda dim, output_size: eval(f"nn.AdaptiveAvgPool{dim}d")(
        output_size = output_size
    ),
    # Converting Multi-Dimensional Feature Maps to 1d Feature Vectors
    "fl_1_-1": lambda start_dim, end_dim: nn.Flatten(
        start_dim=start_dim, end_dim=end_dim
    ),
    "cat_1": lambda dim: lambda i, o: lambda input,dim=dim: torch.cat(input,dim=dim),
    # nn模块，()传递所有参数
    "nn.Linear_(10,1)": lambda func, args: eval(f"nn.{func}")(*args),
}

if __name__ == "__main__":
    import doctest

    doctest.testmod()
