# Linear实现和优化

## 1.基于Linear的语音增强网络实现

### 1.1 IRM语音增强网络

设计一个简单的IRM语音增强网络。

```python
self.mask = torch.nn.Linear(257, 257)
self.activation = torch.nn.Sigmoid()

mask = self.mask(mix_mag)  # (B,T,F)
mask = self.activation(mask)
```

经过训练后，将参数保存为onnx格式，导出的时候需要将Batch这个维度忽略。

```python
x = torch.rand(10, 257)
x = x.to(device)

#define input and output nodes, can be customized
input_names = ["x"]
output_names = ["y"]
#convert pytorch to onnx
torch_out = torch.onnx.export(model, x, "ncnn_test.onnx", input_names=input_names, output_names=output_names)
```

使用netron打开查看。

![](img/Linear实现与优化/model.png ':size=20%')

### 1.2 NCNN中运行

将模型的onnx格式转成ncnn格式。

```shell
./onnx2ncnn ncnn_test.onnx ncnn_test.param ncnn_test.bin
```

param文件内容如下，用另外一个形式表述了网络架构。

```shell
7767517
3 3
Input            x                        0 1 x
InnerProduct     Gemm_0                   1 1 x 3 0=257 1=1 2=66049
Sigmoid          Sigmoid_1                1 1 3 y
```

在NCNN源码中加入模型测试工程：

```shell
# 顶层CMakeLists
add_subdirectory(my_examples)
```

my_examples目录下添加自己写的测试代码和CMakeLists。

```shell
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../src)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/../src)

add_executable(ncnn_test ncnn_test.cpp pffft.c window.c)
target_link_libraries(ncnn_test PRIVATE ncnn)

# add test to a virtual project group
set_property(TARGET ncnn_test PROPERTY FOLDER "my examples")
```

NCNN读取网络和参数，进行计算，关键代码如下：

```cpp
ncnn::Mat in = ncnn::Mat(257, NUM_BLOCK);

ncnn::Net net;
net.load_param("./my_examples/ncnn_test.param");
net.load_model("./my_examples/ncnn_test.bin");
ncnn::Extractor ex = net.create_extractor();
ex.set_light_mode(true);
ex.set_num_threads(4);

ex.input("x", in);
ncnn::Mat feat;
ex.extract("y", feat);
```

编译过程中出现错误，原因是树莓派中libc.so的软链接未链接到任何地方，修改NCNN源码：

```cpp
void* libc_handle = dlopen("libc.so.6", RTLD_NOW);
```

## 2.Linear实现

借助NCNN可以加快深度学习模型的部署，但是转换、移植和调试起来相对会困难一些。NCNN的核心内容就是各种OP的实现，
下面将对NCNN的Linear OP进行分析和抽取，从pytorch直接到嵌入式实现。

### 2.1 pytorch Linear

[pytorch Linear官方文档](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html)

从Linear官方文档可以知道，Linear的参数为`Linear.weight`和`Linear.bias`。

先来打印一下Linear的参数大小。

```python
print(model.mask.weight.shape)
print(model.bias.weight.shape)

#torch.Size([257, 257])
#torch.Size([257])
```




## 3.Linear优化

pytorch linear dump脚本 

导出参数 中间交换格式
linear c语言实现 
arm优化实现 mat pack










