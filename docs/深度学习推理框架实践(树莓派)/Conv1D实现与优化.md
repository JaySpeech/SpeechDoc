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

```powershell
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

`Conv1D`结构体初始化：

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

#### 2.3.1 计算流程

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

`padding`改变输入长度，当`padding=2`时，左右各添加2个数值，长度`1000->1004`。

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
        // 每次循环需要经过outw次(1000)滑动计算，每个out_channel对应256个卷积核，长度为1
        for (int j = 0; j < outw; j++) {
            float sum = 0.f;

            if (conv1d->bias_used == true)
                sum = ((float *)conv1d->bias_mat->data)[p];

            // 卷积权重(257,256,1)，每次读取(256,1)
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

#### 2.3.2 Conv1D NEON优化

* 假设输入`(256,1000)`，输出`(257,1000)`，权重大小`(257,256,1)`，可以优化的地方如下：

```python
以out channel 257为外循环，每次取256个卷积核，大小为1。
将输入(256,1000)和卷积核(256,1)进行滑动乘法，并将in_channels(256)进行叠加 --> 得到 (1000)  
(这里可以进行NEON优化，256可以被4整除)
进行257次外循环后，得到(257,1000)
```

进行计算之前，需要对输入数据`(256,1000)`进行packing操作，packing后大小为`(64,1000)`。<br/>
同样，卷积核`(256,1)`也要进行packing操作，packing后大小为`(64,1)`。

```c
if (elempack == 4 && out_elempack == 1) {
    #pragma omp parallel for num_threads(NUM_THREADS)
    // 处理循环 out_channels 257
    for (int p = 0; p < outh; p++) {
        float* outptr = (float *)Mat_row(conv1d->out4_mat,p);
        // 每次循环需要经过outw次(1000)滑动计算，每个out_channel对应256个卷积核，长度为1
        for (int j = 0; j < outw; j++) {
            float sum = 0.f;

            if (conv1d->bias_used == true) {
                sum = ((float *)conv1d->bias_mat->data)[p];
            }

            // 卷积权重(257,256,1)，每次读取(256,1)，这里是packing后的权重
            const float* kptr = (float *)Mat_channel(conv1d->weight_data_packed,p);

            // h in_channels/4 64
            for (int q = 0; q < h; q++) {
                const float* sptr = (float *)Mat_row(conv1d->input_bordered,q) + j * conv1d->stride_w * 4;

                for (int k = 0; k < conv1d->kernel_w; k++) {
                    float32x4_t _val = vld1q_f32(sptr); // 输入取连续4个数据
                    float32x4_t _w = vld1q_f32(kptr);   // 权重取连续4个数据
                    float32x4_t _s4 = vmulq_f32(_val, _w);
                    #if __aarch64__
                    sum += vaddvq_f32(_s4); // dot
                    #else
                    // 对neon四个数据进行求和
                    float32x2_t _ss = vadd_f32(vget_low_f32(_s4), vget_high_f32(_s4));
                    _ss = vpadd_f32(_ss, _ss);
                    sum += vget_lane_f32(_ss, 0);
                    #endif
                    sptr += conv1d->dilation_w * 4;
                    kptr += 4;
                }
            }
            outptr[j] = sum;
        }
    }
}
```

* 假设输入`(257,1000)`，输出`(256,1000)`，权重大小`(256,257,1)`，可以优化的地方如下：

```python
以out channel为外循环256，每次取257个卷积核，大小为1。
将输入(257,1000)和卷积核(257,1)进行滑动乘法，并将in_channels(257)进行叠加 --> 得到 (1000)
进行256次外循环后，得到(256,1000)
-->
计算之前需要对卷积核(256,257,1)进行一次packing操作，packing后大小为(64,257,1)，也就是每个数据块包含了4个数据
以out channel为外循环64，每次取257个卷积核，大小为1，每个数据块包含了4个数据
将输入(257,1000)(1)和卷积核(257,1)(4)进行滑动乘法，并将in_channels(257)(1)进行叠加 --> 得到 (1000)(4)
(1000次, 每次取1个输入,4个权重，进行并行乘加操作，经过257次完成)
进行64次循环后，得到(64,1000)，每个数据块包含了4个数据。
```

相关代码如下：

```c
if (elempack == 1 && out_elempack == 4) {
    #pragma omp parallel for num_threads(NUM_THREADS)
    // outh 64 外循环
    for (int p = 0; p < outh; p++) {
        float* outptr = (float *)Mat_row(conv1d->out4_mat,p);
        // outw 1000 卷积移动次数
        for (int j = 0; j < outw; j++) {
            float32x4_t _sum = vdupq_n_f32(0.f);

            if (conv1d->bias_used == true) {
                _sum = vld1q_f32((const float*)conv1d->bias_mat->data + p * 4);
            }

            // 取 1 channel权重数据，权重数据大小为(257,1)，里面包含4个数据
            const float* kptr = (float *)Mat_channel(conv1d->weight_data_packed,p);
            // h 257
            for (int q = 0; q < h; q++) {
                const float* sptr = (float *)Mat_row(conv1d->input_bordered,q) + j * conv1d->stride_w;

                for (int k = 0; k < conv1d->kernel_w; k++) {
                    float32x4_t _val = vdupq_n_f32(sptr[0]); // 输入数据复制成4份
                    float32x4_t _w = vld1q_f32(kptr); // 连续取4个权重
                    _sum = vmlaq_f32(_sum, _val, _w);

                    sptr += conv1d->dilation_w;
                    kptr += 4;
                }
            }
            vst1q_f32(outptr, _sum);
            outptr += 4;
        }
    }
}
```

* 假设输入`(256,1000)`，输出`(256,1000)`，权重大小`(256,256,1)`，可以优化的地方如下：

```python
以out channel为外循环256，每次取256个卷积核，大小为1。
将输入(256,1000)和卷积核(256,1)进行滑动乘法，并将in_channels(256)进行叠加 --> 得到 (1000)
进行256次外循环后，得到(256,1000)
-->
计算之前需要对卷积核`(256,256,1)`进行一次packing操作，packing后大小为`(64,64,1)`，每个数据块包含了16个数据
以out channel为外循环64，每次取64个卷积核，大小为1，每个数据块包含了16个数据
将输入(64,1000)(4)和卷积核(64,1)(16)进行滑动乘法，并将in_channels(64)(4)进行叠加 --> 得到(1000)(4)
(1000次, 每次取4个输入，16个权重，一个输入需要寻找对应4个权重，连续进行4次并行乘加操作，经过64次完成)
进行64次循环后，得到(64,1000)
```

> 因为输入数据进行了packing操作，所以对应的卷积核中间的256维度也需要对应进行packing。

相关代码如下：

```c
if (elempack == 4 && out_elempack == 4) {
    #pragma omp parallel for num_threads(NUM_THREADS)
    // outh 64 外循环
    for (int p = 0; p < outh; p++) {
        float* outptr = (float *)Mat_row(conv1d->out4_mat,p);
        // outw 1000 卷积移动次数
        for (int j = 0; j < outw; j++) {
            float32x4_t _sum = vdupq_n_f32(0.f);

            if (conv1d->bias_used == true) {
                _sum = vld1q_f32((const float*)conv1d->bias_mat->data + p * 4);
            }

            const float* kptr = (float *)Mat_channel(conv1d->weight_data_packed,p);
            // h 64 内循环
            for (int q = 0; q < h; q++) {
                const float* sptr = (float *)Mat_row(conv1d->input_bordered,q) + j * conv1d->stride_w * 4;

                for (int k = 0; k < conv1d->kernel_w; k++) {
                    float32x4_t _val = vld1q_f32(sptr);     // 取4个输入数据，前面的优化每次只读一个

                    float32x4_t _w0 = vld1q_f32(kptr);      // 输入数据0对应的4个外循环权重
                    float32x4_t _w1 = vld1q_f32(kptr + 4);  // 输入数据1对应的4个外循环权重
                    float32x4_t _w2 = vld1q_f32(kptr + 8);  // 输入数据2对应的4个外循环权重
                    float32x4_t _w3 = vld1q_f32(kptr + 12); // 输入数据3对应的4个外循环权重

                    // 原来需要256次才能完成所有乘加，现在64次就可以完成，一次循环完成原来4次循环的操作。
                    #if __aarch64__
                    _sum = vmlaq_laneq_f32(_sum, _w0, _val, 0);
                    _sum = vmlaq_laneq_f32(_sum, _w1, _val, 1);
                    _sum = vmlaq_laneq_f32(_sum, _w2, _val, 2);
                    _sum = vmlaq_laneq_f32(_sum, _w3, _val, 3);
                    #else
                    _sum = vmlaq_lane_f32(_sum, _w0, vget_low_f32(_val), 0);
                    _sum = vmlaq_lane_f32(_sum, _w1, vget_low_f32(_val), 1);
                    _sum = vmlaq_lane_f32(_sum, _w2, vget_high_f32(_val), 0);
                    _sum = vmlaq_lane_f32(_sum, _w3, vget_high_f32(_val), 1);
                    #endif

                    sptr += conv1d->dilation_w * 4;
                    kptr += 16;
                }
            }
            vst1q_f32(outptr, _sum);
            outptr += 4;
        }
    }
}
```

## 3.Depthwise Conv1D

在语音处理的场景中，输入channels和输出channels一般相等，等于`FFT(257)`大小。

普通Conv1D，卷积核大小为`(257,257,1)`，也就是每个频带的输出和所有的输入频带相关，
但是实际上不同频带之间的关联性并没有那么强，也就是卷积核可以缩减为`(257,1)`，频带之间保持独立性。

改变`conv1d`中的`groups`可以控制卷积核的变化。

```python
self.mask = torch.nn.Conv1d(257, 257, 1, stride=1, padding=0, dilation=1, groups=257, bias=True)
```

![](img/Conv1D实现与优化/depthwise_conv1d.png ':size=20%')

假设输入`(257,1000)`，输出`(257,1000)`，对比Conv1D和Depthwise Conv1D计算过程：

Conv1D的计算过程：

```c
以out channel为外循环257，每次取257个卷积核，大小为1。
将输入(257,1000)和卷积核(257,1)进行滑动乘法，并将in_channels(257)进行叠加 --> 得到 (1000)
进行257次外循环后，得到(257,1000)
```

Depthwise Conv1D`(groups=257)`计算过程：

```
以groups为外循环257，每次取1个卷积核，大小为1。
将对应的输入(1,1000)和对应的卷积核(1,1)进行滑动乘法 --> 得到(1000)
(也就是输出的每一个频带只和对应的一个输入频带有关系)
进行257次外循环后，得到(257,1000)
```

相关代码如下：

```c
// depth-wise
if (h == conv1d->groups && conv1d->groups == outh) {
    #pragma omp parallel for num_threads(NUM_THREADS)
    // groups 257
    for (int g = 0; g < conv1d->groups; g++) {
        float* outptr = (float *)Mat_row(conv1d->out_mat,g); // 输出(257,1000)取一行(1,1000)
        const float* kptr = (const float*)conv1d->weight_mat->data + conv1d->kernel_w * g; // 从(257,1)取对应的权重(1,1)

        // outw 1000
        for (int j = 0; j < outw; j++) {
            float sum = 0.f;

            if (conv1d->bias_used == true)
                sum = ((float *)conv1d->bias_mat->data)[g];
            
            const float* sptr = (float *)Mat_row(conv1d->input_bordered,g) + j * conv1d->stride_w; // 输入(257,1000)取一行，并根据j进行滑动

            for (int k = 0; k < conv1d->kernel_w; k++) {
                float val = *sptr;
                float w = kptr[k];
                sum += val * w;

                sptr += conv1d->dilation_w;
            }

            outptr[j] = sum;
        }
    }
}
```

Depthwise Conv1D无法进行Neon并行计算：
* kernel size一般为3，不满足Neon要求
* 输入和卷积核都是取其中的一行进行计算，不满足Neon要求


## 4.Groups Conv1D

Depthwise Conv1D属于Groups Conv1D的一个特殊情况，如下设置为Groups Conv1D。

```python
self.mask = torch.nn.Conv1d(256, 256, 1, stride=1, padding=0, dilation=1, groups=128, bias=True)
```

计算流程如下：

```c
卷积核大小为(256,2,1)
以groups为外循环128，每次取4个卷积核，大小为1。
将对应的输入(2,1000)和对应的卷积核(4,1)进行滑动乘法,并相加 --> 得到(2,1000)
进行128次外循环后，得到(256,1000)
```

相关代码如下：

```c
// group convolution
const int h_g = h / conv1d->groups;             // 2
const int outh_g = outh / conv1d->groups;       // 2

#pragma omp parallel for collapse(2) num_threads(NUM_THREADS)
// groups 128
for (int g = 0; g < conv1d->groups; g++) {
    // outh_g 2 每次计算两个输出(2,1000)
    for (int p = 0; p < outh_g; p++) {
        // 计算一行输出(1,1000)
        float* outptr = (float *)Mat_row(conv1d->out_mat, g * outh_g + p); // g = 0 (0,1) g = 1 (2,3) ... g = 127 (255,256)
        // 从权重(256,2,1)取对应的权重，g = 0 偏移0， g = 1 偏移 4, g = 2 偏移 8
        // 两个输出需要4组权重
        const float* weight_data_ptr = (const float*)conv1d->weight_mat->data + conv1d->kernel_w * h_g * outh_g * g;

        // outw 1000
        for (int j = 0; j < outw; j++) {
            float sum = 0.f;

            if (conv1d->bias_used == true)
                sum = ((float *)conv1d->bias_mat->data)[outh_g * g + p];

            const float* kptr = weight_data_ptr + conv1d->kernel_w * h_g * p; // 取权重(2,1)

            // 取两行输入(2,1000)和权重(2,1)进行卷积计算并求和，输出(1000)
            for (int q = 0; q < h_g; q++) {
                const float* sptr = (float *)Mat_row(conv1d->input_bordered, h_g * g + q) + j * conv1d->stride_w;

                for (int k = 0; k < conv1d->kernel_w; k++) {
                    float val = *sptr;
                    float w = kptr[k];
                    sum += val * w;

                    sptr += conv1d->dilation_w;
                }

                kptr += conv1d->kernel_w;
            }

            outptr[j] = sum;
        }
    }
}
```

## 5.运行结果

```powershell
Conv1D
(256,1000) -> (257,1000)
normal 449616us arm_1 347188us arm_2 195457us arm_4 122237us

Depthwise Conv1D
(257,1000) -> (257,1000)
normal 19819us normal_2 15357us norm_4 9270us
```