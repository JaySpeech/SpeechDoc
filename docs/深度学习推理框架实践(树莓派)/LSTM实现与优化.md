# LSTM实现与优化

## 1.基于LSTM的语音增强网络实现

### 1.1 IRM语音增强网络

设计一个简单的IRM语音增强网络。

```python
self.mask = torch.nn.LSTM(256, 257, num_layers=1, bidirectional=False, batch_first=False)
self.activation = torch.nn.Sigmoid()

mask = self.mask(mix_mag)  # (B,T,F)
mask = self.activation(mask)
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

使用netron打开查看，网络结构中多了好多奇怪的分支。

![](img/LSTM实现与优化/lstm_onnx.png ':size=30%')

使用`onnx-simplifier`进行简化。

```shell
pip install onnx-simplifier
python -m onnxsim ncnn_test.onnx ncnn_test_op.onnx 
```

简化后的网络结构。

![](img/LSTM实现与优化/lstm_onnx_op.png ':size=25%')


### 1.2 NCNN中运行

将模型的onnx格式转成ncnn格式。

param文件内容如下，用另外一个形式表述了网络架构。

```shell
7767517
3 5
Input            x                        0 1 x
LSTM             LSTM_9                   1 3 x 46 44 45 0=257 1=263168 2=0
Sigmoid          Sigmoid_11               1 1 46 y
```
![](img/LSTM实现与优化/lstm_ncnn.png ':size=25%')


## 2.LSTM实现

### 2.1 LSTM基本原理

![](img/LSTM实现与优化/lstm_框架图.png ':size=50%')

LSTM的关键就是细胞状态，水平线在图上方贯穿运行。

![](img/LSTM实现与优化/lstm_00.png ':size=50%')

LSTM 有通过精心设计的称作为`门`的结构来去除或者增加信息到细胞状态的能力。门是一种让信息选择式通过的方法。他们包含一个`sigmoid`神经网络层和一个按位的乘法操作。

Sigmoid 层输出`0`到`1`之间的数值，描述每个部分有多少量可以通过。`0`代表“不许任何量通过”，`1`就指“允许任意量通过”。

![](img/LSTM实现与优化/lstm_门.png ':size=10%')

LSTM 拥有三个门，来保护和控制细胞状态。

* 忘记门，决定我们会从细胞状态中丢弃什么信息。

![](img/LSTM实现与优化/lstm_01.png ':size=50%')

* 更新门，确定什么样的新信息被存放在细胞状态中。

    这里包含两个部分。第一，sigmoid 层称 “输入门层” 决定什么值我们将要更新。
    然后，一个 tanh 层创建一个新的候选值向量，$\tilde{C}_t$，会被加入到状态中。下一步，我们会讲这两个信息来产生对状态的更新。

![](img/LSTM实现与优化/lstm_02.png ':size=50%')

* 更新细胞状态

    我们把旧状态与$f_t$相乘，丢弃掉我们确定需要丢弃的信息。接着加上$i_t * \tilde{C}_t$。这就是新的候选值，根据我们决定更新每个状态的程度进行变化。

![](img/LSTM实现与优化/lstm_03.png ':size=50%')

* 输出门
    
    最终，我们需要确定输出什么值。

![](img/LSTM实现与优化/lstm_04.png ':size=50%')


### 2.2 pytorch LSTM和权重导出

[pytorch LSTM官方文档](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)

从LSTM官方文档可以知道，LSTM的参数为`weight_ih_l[k]`,`weight_hh_l[k]`,`bias_ih_l[k]`,`bias_hh_l[k]`。

$$
\begin{aligned}
i_{t} &=\sigma\left(W_{i i} x_{t}+b_{i i}+W_{h i} h_{t-1}+b_{h i}\right) \\
f_{t} &=\sigma\left(W_{i f} x_{t}+b_{i f}+W_{h f} h_{t-1}+b_{h f}\right) \\
g_{t} &=\tanh \left(W_{i g} x_{t}+b_{i g}+W_{h g} h_{t-1}+b_{h g}\right) \\
o_{t} &=\sigma\left(W_{i o} x_{t}+b_{i o}+W_{h o} h_{t-1}+b_{h o}\right) \\
c_{t} &=f_{t} \odot c_{t-1}+i_{t} \odot g_{t} \\
h_{t} &=o_{t} \odot \tanh \left(c_{t}\right)
\end{aligned}
$$

先来打印一下LSTM的参数大小。

```python
print(model.mask.weight_ih_l0.shape)
print(model.mask.weight_hh_l0.shape)
print(model.mask.bias_ih_l0.shape)
print(model.mask.bias_hh_l0.shape)

#torch.Size([1028, 256])
#torch.Size([1028, 257])
#torch.Size([1028])
#torch.Size([1028])
```

