"""
>>> from monet.torch_ddf import torch_dict
>>> Fc = torch_dict['fc_True'](True)(2,1)
>>> Fc
Linear(in_features=2, out_features=1, bias=True)
"""

import torch.nn as nn

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
    "act.relu_()": lambda activation, args: eval(f"nn.{activation}")(*args),
    # Common activation functions
    "relu_b": lambda b=False: nn.ReLU(inplace=b),
    "prelu_p": lambda p=0.25: nn.PReLU(init=p),
    "gelu_s": lambda s="none": nn.GELU(str=s),
    "softmax_d": lambda d=None: nn.Softmax(dim=d),
    "tanh": nn.Tanh(),
    "sigmoid": nn.Sigmoid(),
    # DropOut, DropOutNdim, AlphaDropout, FeatureAlphaDropout
    # bouth have p and innplace option
    "dp_0.5_False": lambda p, inplace: lambda i, o: nn.Dropout(
        p=p, inplace=bool(inplace)
    ),
    "dp1_1_False": lambda Ndim, p, inplace: lambda i, o: eval(f"nn.Dropout{Ndim}d")(
        p=p, inplace=bool(inplace)
    ),
    # AlphaDropOut
    "adp_0.5_False": lambda p, inplace: lambda i, o: nn.AlphaDropout(
        p=p, inplace=bool(inplace)
    ),
    "fadp_0.5_False": lambda p, inplace: lambda i, o: nn.FeatureAlphaDropout(
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
    "lrn_": lambda i, o: lambda i, o: nn.LocalResponseNorm(size=i),
    # N-dimensional convolution
    "cv2_3_1": lambda Ndim, kernel_size, stride: lambda i, o: eval(f"nn.Conv{Ndim}d")(
        in_channels=i, out_channels=o, kernel_size=kernel_size, stride=stride
    ),
    # Transposed convolution
    "cvT2_3_1": lambda Ndim, kernel_size, stride: lambda i, o: eval(
        f"nn.ConvTranspose{Ndim}d"
    )(in_channels=i, out_channels=o, kernel_size=kernel_size, stride=stride),  # 反卷积
    # N-dimensional Maximum pooling
    "mp2_2_1": lambda dim, kernel_size, stride: eval(f"nn.MaxPool{dim}d")(
        kernel_size=kernel_size, stride=stride
    ),
    # Adaptive N-dimensional Maximum pooling
    "amp2_2_0": lambda dim, kernel_size, stride: eval(
        f"nn.AdaptiveMaxPool{dim}d"
    )(kernel_size=kernel_size, stride=stride),  # 自适应最大池化
    # N-dimensional Avarage pooling
    "ap2_1": lambda dim, stride: lambda i, o: eval(f"nn.AvgPool{dim}d")(
        padding=stride
    ),
    # Adaptive N-dimensional Avarage pooling
    "aap2_1": lambda dim, stride: lambda i, o: eval(f"nn.AdaptiveAvgPool{dim}d")(
        stride=stride
    ),
    # Converting Multi-Dimensional Feature Maps to 1d Feature Vectors
    "fl_1_-1": lambda start_dim, end_dim: nn.Flatten(
        start_dim=start_dim, end_dim=end_dim
    ),
    # nn模块，()传递所有参数
    "nn.Linear_(10,1)": lambda func, args: eval(f"nn.{func}")(*args),
}

if __name__ == "__main__":
    import doctest

    doctest.testmod()
