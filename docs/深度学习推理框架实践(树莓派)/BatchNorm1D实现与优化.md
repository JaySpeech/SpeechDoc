# BatchNorm1D实现和优化

## 1.基于BatchNorm1D的语音增强网络实现

### 1.1 IRM语音增强网络

设计一个简单的IRM语音增强网络。

```python
self.norm = torch.nn.BatchNorm1d(257)
self.activation = torch.nn.PReLU()

mix_mag = mix_mag.permute(0,2,1)
mask = self.norm(mix_mag)  # (B,T,F)
mask = self.activation(mask)
mask = mask.permute(0,2,1)
```

经过训练后，将参数保存为onnx格式。

```python
x = torch.rand(1, 10, 256)
x = x.to(device)

#define input and output nodes, can be customized
input_names = ["x"]
output_names = ["y"]
#convert pytorch to onnx
torch_out = torch.onnx.export(model, x, "ncnn_test.onnx", input_names=input_names, output_names=output_names)
```

使用netron打开查看：

![](img/BatchNorm1D实现与优化/bn_onnx.png ':size=20%')

### 1.2 NCNN中运行

将模型的onnx格式转成ncnn格式。

param文件内容如下，用另外一个形式表述了网络架构。


```powershell
7767517
5 5
Input            x                        0 1 x
Permute          Transpose_0              1 1 x 9 0=1
BatchNorm        BatchNormalization_1     1 1 9 10 0=257
PReLU            PRelu_2                  1 1 10 12 0=1
Permute          Transpose_3              1 1 12 y 0=1
```

![](img/BatchNorm1D实现与优化/bn_ncnn.png ':size=20%')

## 2.BatchNorm1D实现

### 2.1 BatchNorm1D原理

Batch Normalization （以下简称为 BN ）来自两位 Google 研究员发表的一篇重要论文[Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shif](https://arxiv.org/abs/1502.03167) ，中文一般翻译为“批标准化/规范化”。其核心思想是，在深度网络的中间层内添加正态标准化处理（作为 BN 层出现），同时约束网络在训练过程中自动调整该标准化的强度，从而加快训练速度并降低权值初始化的成本。 

$$
\mu_B = \frac{1}{B} \sum_i^B \mu_i ，\\
\sigma_B^2 = \frac{1}{B - 1} \sum_i^B \sigma^2_i ，\\
\hat{X} = \frac{X - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}\\
$$

做完`BN`之后，还需要将其还原： 

$$
y = \gamma \hat{X} + \beta
$$

$\gamma$和$\beta$是自适应的参数，通常设定初始值$\gamma=1,\beta=0$。


下面将使用python实现`BatchNorm1D`的前向传播过程：

> 这里未关注`running_mean`和`running_var`实现。

```python
import torch

def fowardbn(x, gam, beta, ):
    '''
    momentum = 0.1
    running_mean = 0
    running_var = 1
    running_mean = (1 - momentum) * running_mean + momentum * x.mean(dim=0)
    running_var = (1 - momentum) * running_var + momentum * x.var(dim=0)
    '''
    eps = 1e-05
    mean = x.mean(dim=0)
    var = x.var(dim=0,unbiased=False)
    x_hat = (x - mean) / torch.sqrt(var + eps)
    out = gam * x_hat + beta
    cache = (x, gam, beta, x_hat, mean, var, eps)
    return out, cache

model2 = torch.nn.BatchNorm1d(5,affine=True)
input1 = torch.randn(3, 5, requires_grad=True)
input2 = input1.clone().detach().requires_grad_()
x = model2(input1)
out, cache = fowardbn(input2, model2.weight, model2.bias)
#print(x,out)
```


### 2.2 pytorch BatchNorm1D权重导出

[pytorch BatchNorm1d官方文档](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html)


```python
torch.nn.BatchNorm1d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None)
```

