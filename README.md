# MoNet

一个新型神经网络AI表述与实现框架，基于pytorch，scikit-learn，skorch，实现网络的公式化表达与构建。网络列表如下：

| 公式        | 带参公式               | 带参代码         | 含义           | 实现                                                                                                                              |
| ----------- | ---------------------- | ---------------- | -------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| $Fc$      | $Fc_{true}^{o}$      | fc_True          | 全连接层       | `nn.Linear(in_features=i,out_features=o,bias=bias)`                                                                             |
| $Bfc$     | $Bfc_{true}^{o}$     | bfc_True         | 双线性层       | `nn.Bilinear(in1_features=i[0],in2_features=i[1],out_features=o,bias=bias)`                                                     |
| $Fl   $   | $Fl_{1,-1}^o$        | flat_1_-1        | 压扁层         | `nn.Flatten(start_dim=start_dim,end_dim=end_dim)`                                                                               |
| $Cv     $ | $Cv2_{3,1,0,true}^o$ | cov2_3_1_0_True  | 卷积层         | `eval(f"nn.Conv{dim}d")(in_channels=i,out_channels=o,kernel_size=kernel_size,`                                                  |
| $Cvt$     | $Cvt2$               | covT2_3_1_0_True | 反卷积层       | `eval(f"nn.ConvTranspose{dim}d")(in_channels=i,out_channels=o,kernel_size=kernel_size,stride=stride,padding=padding,bias=bias)` |
| $Mp$      | $Mp2$                | mp2_2_0          | 最大池化       | `eval(f"nn.MaxPool{dim}d")(kernel_size=kernel_size,padding=padding)`                                                            |
| $Amp$     | $Mpa2$               | amp2_2_0         | 自适应最大池化 | `eval(f"nn.AdaptiveMaxPool{dim}d")(kernel_size=kernel_size,padding=padding)`                                                    |
| $Ap$      | $Ap2$                | ap2_2            | 平均池化       | `eval(f"nn.AvgPool{dim}d")(padding=padding)`                                                                                    |
| $Aap $    | $Apa2$               | aap2_2           | 自适应平均池化 | `eval(f"nn.AdaptiveAvgPool{dim}d")(padding=padding)`                                                                            |
| $Bn  $    | $Bn1$                | bn1_0            | 批归一化层     | `eval(f"nn.BatchNorm{dim}d")(num_features=i if num_features==0else num_features)`                                               |
| $In$      | $In1$                | in1_0            | 归一化层       | `eval(f"nn.InstanceNorm{dim}d")(num_features=i if num_features==0else num_features)`                                            |
| $Gn   $   | $Gn$                 | gn_2             | 组归一化层     | `nn.GroupNorm(num_groups=num_groups,num_channels=i)`                                                                            |
| $Ln $     | $Ln$                 | ln               | 归一化层       | `nn.LayerNorm(normalized_shape=i)`                                                                                              |
| $Lrn   $  | $Lrn$                | lrn              | 归一化层       | `nn.LocalResponseNorm(size=i)`                                                                                                  |
| $Dp    $  | $Dp$                 | dp_0.5_False     | 隐藏层         | `nn.Dropout(p=p,inplace=bool(inplace))`                                                                                         |
| $Dp   $   | $Dp1$                | dp1_1_False      | 隐藏层         | `eval(f"nn.Dropout{dim}d")(p=p,inplace=bool(inplace))`                                                                          |
| $Adp    $ | $Adp$                | aldp_0.5_False   | 隐藏层         | `nn.AlphaDropout(p=p,inplace=bool(inplace))`                                                                                    |
| $Fadp  $  | $Fadp$               | fadp_0.5_False   | 隐藏层         | `nn.FeatureAlphaDropout(p=p,inplace=bool(inplace))`                                                                             |
| $Act  $   | $Act$                | act.PReLU        | 激活层         | eval(f"nn.{act_func}")()                                                                                                          |
| $Nn $     | $Nn$                 | nn.Linear_(10,1) | 通配层         | eval(f"nn.{func}")(*args)                                                                                                         |

## 如何用公式表达一个网络

### Lenet

![1700822677574](image/README/1700822677574.png)

![1700834361797](image/README/1700834361797.jpg)

$$
\begin{array}{c}
Lenet = CvSpCvSpFcFcGc = 2(CvSp)2FcGc_{}\\
Lenet_{32×32}^{1} = Cv2_{5}^6Sp2_{2}^6Cv2_{5}^{16}Sp2_{2}^{16}Fc_{}^{120}Fc_{}^{84}Gc_{}^{10}\\ = (Cv2_{5}Sp2_{2})_{}^{[6,16]}Fc_{}^{[120,84]}Gc_{}^{10}\\ = (Cv_{5}Sp_{2})2_{}^{[6,16]}Fc_{}^{[120,84]}Gc_{}^{10}\\ = (Cv_{5}Sp_{2})_{}^{[6,16]}Fc_{}^{[120,84]}Gc_{}^{10}
\end{array}
$$


# 代码实现

```python
import monet as mn
Lenet = mn.Mix(1, [[6,16],[120,64],10], [['cv_5','sp_2'],'fc','gc'])
```
