# Linear实现和优化

## 1.基于Linear的语音增强网络实现

### 1.1 IRM语音增强网络

设计一个简单的IRM语音增强网络。

```python
self.mask = torch.nn.Linear(256, 257)
self.activation = torch.nn.Sigmoid()

mask = self.mask(mix_mag)  # (B,T,F)
mask = self.activation(mask)
```

经过训练后，将参数保存为onnx格式，导出的时候需要将Batch这个维度忽略。

```python
x = torch.rand(10, 256)
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
InnerProduct     Gemm_0                   1 1 x 3 0=257 1=1 2=65792
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
ncnn::Mat in = ncnn::Mat(256, NUM_BLOCK);

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

### 2.1 pytorch Linear和权重导出

[pytorch Linear官方文档](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html)

从Linear官方文档可以知道，Linear的参数为`Linear.weight`和`Linear.bias`。

先来打印一下Linear的参数大小。

```python
print(model.mask.weight.shape)
print(model.bias.weight.shape)

#torch.Size([256, 257])
#torch.Size([257])
```

权重直接导出为二进制文件。

```python
if os.path.exists("mask_param.bin"):
        os.remove("mask_param.bin")

mask_weight = np.float32(model.mask.weight.detach().cpu().numpy())
mask_weight = list(mask_weight.reshape(-1))
print("mask_weight len:" + str(len(mask_weight)))
data = struct.pack('f'*len(mask_weight),*mask_weight)
with open("mask_param.bin",'ab+') as f:
    f.write(data)

mask_bias = np.float32(model.mask.bias.detach().cpu().numpy())
mask_bias = list(mask_bias.reshape(-1))
print("mask_bias len:" + str(len(mask_bias)))
data = struct.pack('f'*len(mask_bias),*mask_bias)
with open("mask_param.bin",'ab+') as f:
    f.write(data)
```

### 2.2 Linear权重导入和C语言实现

Linear结构体初始化：

```c
Linear *Linear_create(int in_features, int out_features, bool bias_used, int _elemsize){
    Linear* linear = (Linear *)malloc(sizeof(Linear));
    if(linear == NULL){
        return NULL;
    }

    linear->in_features = in_features;
    linear->out_features = out_features;
    linear->bias_used = bias_used;
    linear->elemsize = _elemsize;

    linear->weight_mat = Mat_2D_create(in_features,out_features,_elemsize);
    linear->bias_mat = Mat_1D_create(out_features,_elemsize);

    return linear;
}
```

全部中导入方法如下，二进制传递参数比较直接，后面可以考虑采用结构化数据进行参数传递。

```c
int Linear_load_variables(Linear *linear, char* file){
    if(linear == NULL){
        return -1;
    }

    FILE * mask_bin_file = fopen("./mask_param.bin","rb");
    if(mask_bin_file == NULL){
        return -1;
    }

    fread(linear->weight_mat->data,sizeof(float),linear->in_features*linear->out_features,mask_bin_file);
    //Mat_2D_float_printf(linear->weight_mat);

    if(linear->bias_used == true){
        fread(linear->bias_mat->data,sizeof(float),linear->out_features,mask_bin_file);
        //Mat_1D_float_printf(linear->bias_mat);
    }

    fclose(mask_bin_file);

    return 0;
}
```

Linear为矩阵乘法运算，实现起来比较简单。

```c
int Linear_process(Linear *linear, Mat *input, Mat* output){
    if(linear == NULL){
        return -1;
    }

    if(input->dims == 2 && output->dims ==2 && \
        input->w == linear->in_features && output->w == linear->out_features) {

        #pragma omp parallel for num_threads(NUM_THREADS)
        for (int j = 0; j < input->h; j++) {

            const float* m = (float *)Mat_row(input, j);
            float* outptr = (float *)Mat_row(output, j);

            for (int p = 0; p < linear->out_features; p++) {
                const float* kptr = (const float*)linear->weight_mat->data + input->w * p;

                float sum = 0.f;

                if(linear->bias_used){
                    sum = *((float *)linear->bias_mat->data + p);
                }

                for (int i = 0; i < input->w; i++){
                    sum += m[i] * kptr[i];
                }

                outptr[p] = sum;
            }
        }
        return 0;
    }

    return -1;
}
```

## 3.Linear优化

arm优化实现 mat pack bf16










