# Conv1D实现和优化

## 1.基于Conv1D的语音增强网络实现

### 1.1 IRM语音增强网络

设计一个简单的IRM语音增强网络。

```python
self.mask = torch.nn.Conv1d(256, 257, 1, stride=1, padding=0, dilation=1, groups=1, bias=True)
self.activation = torch.nn.Sigmoid()

mix_mag = mix_mag.permute(0,2,1)
mask = self.mask(mix_mag)  # (B,T,F)
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

![](img/Conv1D实现与优化/model_onnx.png ':size=20%')

### 1.2 NCNN中运行

将模型的onnx格式转成ncnn格式。

param文件内容如下，用另外一个形式表述了网络架构。

```shell
7767517
5 5
Input            x                        0 1 x
Permute          Transpose_0              1 1 x 3 0=1
Convolution1D    Conv_1                   1 1 3 4 0=257 1=1 2=1 3=1 4=0 14=0 5=1 6=65792
Sigmoid          Sigmoid_2                1 1 4 5
Permute          Transpose_3              1 1 5 y 0=1
```

![](img/Conv1D实现与优化/model_ncnn.png ':size=20%')

## 2.Conv1D实现

### 2.1 pytorch Conv1D和权重导出

[pytorch Conv1d官方文档](https://pytorch.org/docs/1.9.1/generated/torch.nn.Conv1d.html)

Conv1D参数大小

```python
print(model.mask.weight.shape)
print(model.mask.bias.shape)

#torch.Size([257, 256, 1])
#torch.Size([257])
```

导出方法和Linear一样：

```python
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

### 2.2 Conv1D权重导入

Linear结构体初始化：

```c
Conv1D *Conv1D_create(int in_channels, int out_channels, int kernel_w, \
                      int dilation_w, int stride_w, int pad_left, \
                      int pad_right, float pad_value, bool bias_used){
    Conv1D* conv1d = (Conv1D *)malloc(sizeof(Conv1D));
    if(conv1d == NULL){
        return NULL;
    }

    conv1d->in_channels = in_channels;
    conv1d->out_channels = out_channels;

    conv1d->kernel_w = kernel_w;
    conv1d->dilation_w = dilation_w;
    conv1d->stride_w = stride_w;

    conv1d->pad_left = pad_left;
    conv1d->pad_right = pad_right;
    conv1d->pad_value = pad_value;

    conv1d->bias_used = bias_used;

    conv1d->weight_mat = Mat_1D_create(in_channels*out_channels*kernel_w,4u,1);

    if(conv1d->bias_used == true){
        conv1d->bias_mat = Mat_1D_create(out_channels,4u,1);
    }

    conv1d->input_bordered = NULL;
    conv1d->out_mat = NULL;

    return conv1d;
}
```

参数导入：

```c
int Conv1D_load_variables(Conv1D *conv1d, char *file){
    if(conv1d == NULL){
        return -1;
    }

    FILE * weight_bin_file = fopen(file,"rb");
    if(weight_bin_file == NULL){
        return -1;
    }

    fread(conv1d->weight_mat->data, sizeof(float), conv1d->in_channels*conv1d->out_channels*conv1d->kernel_w, weight_bin_file);
    //Mat_1D_float_printf(conv1d->weight_mat);

    if(conv1d->bias_used == true){
        fread(conv1d->bias_mat->data, sizeof(float), conv1d->out_channels, weight_bin_file);
        //Mat_1D_float_printf(conv1d->bias_mat);
    }

    fclose(weight_bin_file);

    return 0;
}
```

### 2.3 Conv1D实现

这里假设输入为`(256,1000)`，输出为`(257,1000)`。

卷积计算的大致流程：

```python
#torch.nn.Conv1d(256, 257, 1, stride=1, padding=0, dilation=1, groups=1, bias=True)

输入(256,1000)，输出(257,1000)，权重大小(257,256,1)

以out channel为外循环，每次取256个卷积核，大小为1。
将输入(256,1000)和卷积核(256,1)进行滑动乘法，并将in_channels(256)进行叠加 --> 得到 (1000)
进行257次外循环后，得到(257,1000)
```

Conv1D输出长度计算方法：

$$
L_{\text {out }}=\frac{L_{i n}+2 \times \text { padding }-\text { dilation } \times (\text { kernelsize }-1)-1 }{\text { stride }}+1
$$

#### 2.3.1 Padding



#### 2.3.2 计算流程

下面对Conv1D中的`dilation`和`stride`进行简单分析。

`dilation`引起卷积核扩张。

```c
// 当kernel size为3，dilation为2时，kernel_extent_w = 5   *** -> *o*o*
// 当kernel size为3，dilation为3时，kernel_extent_w = 7   *** -> *oo*oo*
const int kernel_extent_w = conv1d->dilation_w * (conv1d->kernel_w - 1) + 1;
```

`stride`改变卷积步长。

```c
// 当长度w=10，kernel_extent_w=5，stride_w=1，outw = 6
// 当长度w=10，kernel_extent_w=5，stride_w=2，outw = 3
const int outw = (w - kernel_extent_w) / conv1d->stride_w + 1;
```

```c
int Conv1D_process(Conv1D *conv1d, Mat *input){
    if(conv1d == NULL){
        return -1;
    }

    if(conv1d->pad_left > 0 || conv1d->pad_right > 0){
        // 这里还未完善
    }
    else{
        conv1d->input_bordered = input;
    }

    // w = 1000
    const int w = conv1d->input_bordered->w;
    const size_t elemsize = conv1d->input_bordered->elemsize;
    const int kernel_extent_w = conv1d->dilation_w * (conv1d->kernel_w - 1) + 1; //扩张卷积核大小
    const int outw = (w - kernel_extent_w) / conv1d->stride_w + 1; // 输出长度计算

    if(conv1d->out_mat == NULL){
        conv1d->out_mat = Mat_2D_create(outw,conv1d->out_channels,elemsize,1);
    }

    // h padding后卷积数据长度
    const int h = conv1d->input_bordered->h;
    const int outh = conv1d->out_channels;

    #pragma omp parallel for num_threads(NUM_THREADS)
    // 处理循环 out_channels 257
    for (int p = 0; p < outh; p++) {
        float* outptr = (float *)Mat_row(conv1d->out_mat,p);
        // 每次循环需要经过多少次(outw,1000)计算，每个out_channel对应256个卷积核，长度为1
        for (int j = 0; j < outw; j++) {
            float sum = 0.f;

            if (conv1d->bias_used == true)
                sum = ((float *)conv1d->bias_mat->data)[p];

            // 卷积权重(257,256,1)
            // 每次读取(256,1)
            const float* kptr = (const float*)conv1d->weight_mat->data + conv1d->kernel_w * h * p;

            // h in_channels 256
            for (int q = 0; q < h; q++) {
                // 获取一个channel的数据第q次滑动乘法的数据起始位置
                const float* sptr = (float *)Mat_row(conv1d->input_bordered,q) + j * conv1d->stride_w;

                // 卷积乘法
                for (int k = 0; k < conv1d->kernel_w; k++) {
                    float val = *sptr;
                    float wt = kptr[k];
                    sum += val * wt;

                    sptr += conv1d->dilation_w; // 跳过中间的空洞
                }

                kptr += conv1d->kernel_w;
            }

            outptr[j] = sum;
        }
    }

    return 0;
}
```

### 2.3 Conv1D NEON优化




实时性考虑，arm加速对实时性的帮助大吗
1.将conv1d arm优化考虑进来
2.Padding还未完善
3.将groups考虑进来 并做arm优化
4.计算mips groups和非groups的耗时