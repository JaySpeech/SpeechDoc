# BatchNorm1D实现和优化

## 1.基于BatchNorm1D的语音增强网络实现

### 1.1 IRM语音增强网络

设计一个简单的IRM语音增强网络。

```python
self.norm = torch.nn.BatchNorm1d(257)
self.activation = torch.nn.PReLU()

mix_mag = mix_mag.permute(0,2,1) # (B,F,T)
mask = self.norm(mix_mag)  
mask = self.activation(mask)
mask = mask.permute(0,2,1)
```

经过训练后，将参数保存为onnx格式。

```python
x = torch.rand(1, 10, 257)
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

Batch Normalization来自两位 Google 研究员发表的一篇重要论文[Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shif](https://arxiv.org/abs/1502.03167)，中文一般翻译为“批标准化/规范化”。其核心思想是，在深度网络的中间层内添加正态标准化处理（作为BN层出现），同时约束网络在训练过程中自动调整该标准化的强度，从而加快训练速度并降低权值初始化的成本。 

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

下面对`affine`和`track_running_stats`这两个变量进行解析。

`affine=True`表示使用$\gamma$和$\beta$，在训练中会更新这两个参数，反之使用默认值。

`track_running_stats=True`表示使用`running_mean`和`running_mean`，反之使用`mean`和`var`。

当`affine=False`并且`track_running_stats=False`，推理的过程中可以忽略BN层。

```python
print(model.norm.weight.shape)
print(model.norm.bias.shape)
print(model.norm.running_mean.shape)
print(model.norm.running_var.shape)

#torch.Size([257])
#torch.Size([257])
#torch.Size([257])
#torch.Size([257])
```

导出方法为：

```python
norm_weight = np.float32(model.norm.weight.detach().cpu().numpy())
norm_weight = list(norm_weight.reshape(-1))
print("norm_weight len:" + str(len(norm_weight)))
data = struct.pack('f'*len(norm_weight),*norm_weight)
with open("norm_param.bin",'ab+') as f:
    f.write(data)

norm_bias = np.float32(model.norm.bias.detach().cpu().numpy())
norm_bias = list(norm_bias.reshape(-1))
print("norm_bias len:" + str(len(norm_bias)))
data = struct.pack('f'*len(norm_bias),*norm_bias)
with open("norm_param.bin",'ab+') as f:
    f.write(data)

norm_running_mean = np.float32(model.norm.running_mean.detach().cpu().numpy())
norm_running_mean = list(norm_running_mean.reshape(-1))
print("norm_running_mean len:" + str(len(norm_running_mean)))
data = struct.pack('f'*len(norm_running_mean),*norm_running_mean)
with open("norm_param.bin",'ab+') as f:
    f.write(data)

norm_running_var = np.float32(model.norm.running_var.detach().cpu().numpy())
norm_running_var = list(norm_running_var.reshape(-1))
print("norm_running_var len:" + str(len(norm_running_var)))
data = struct.pack('f'*len(norm_running_var),*norm_running_var)
with open("norm_param.bin",'ab+') as f:
    f.write(data)
```

### 2.3 BatchNorm1D权重导入

`BatchNorm1D`结构体初始化：

```c
BatchNorm *BatchNorm_create(int channels){
    BatchNorm* bn = (BatchNorm *)malloc(sizeof(BatchNorm));
    if(bn == NULL){
        return NULL;
    }

    bn->channels = channels;

    bn->slope_data_mat = Mat_2D_create(channels,1,4u,1);
    bn->mean_data_mat = Mat_2D_create(channels,1,4u,1);
    bn->var_data_mat = Mat_2D_create(channels,1,4u,1);
    bn->bias_data_mat = Mat_2D_create(channels,1,4u,1);

    bn->a_data_mat = Mat_1D_create(channels,4u,1);
    bn->b_data_mat = Mat_1D_create(channels,4u,1);

    return bn;
}
```

参数导入：

```c
int BatchNorm_load_variables(BatchNorm *bn, char *file){
    if(bn == NULL){
        return -1;
    }

    FILE * weight_bin_file = fopen(file,"rb");
    if(weight_bin_file == NULL){
        return -1;
    }

    fread(bn->slope_data_mat->data, sizeof(float), bn->channels, weight_bin_file);
    fread(bn->bias_data_mat->data, sizeof(float), bn->channels, weight_bin_file);
    fread(bn->mean_data_mat->data, sizeof(float), bn->channels, weight_bin_file);
    fread(bn->var_data_mat->data, sizeof(float), bn->channels, weight_bin_file);

    float eps = 0.00001;

    for (int i = 0; i < bn->channels; i++) {
        float sqrt_var = sqrt(((float *)bn->var_data_mat->data)[i] + eps);
        
        ((float *)bn->a_data_mat->data)[i] = ((float *)bn->bias_data_mat->data)[i] - \
                                             ((float *)bn->slope_data_mat->data)[i] * ((float *)bn->mean_data_mat->data)[i] / sqrt_var;
        
        ((float *)bn->b_data_mat->data)[i] =  ((float *)bn->slope_data_mat->data)[i] / sqrt_var;
    }

    fclose(weight_bin_file);

    return 0;
}
```

权重导入后，预先对权重进行了处理，避免后续进行多次计算。

$$
y=\frac{x-\mathrm{E}[x]}{\sqrt{\operatorname{Var}[x]+\epsilon}} * \gamma+\beta
$$


### 2.4 BatchNorm1D实现

假设输入为`(257,3)(F,T)`，`w=3`并且`h=257`，参数大小为`(257)`

```c
int dims = input->dims;

if (dims == 2) {
    int w = input->w;
    int h = input->h;

    #pragma omp parallel for num_threads(NUM_THREADS)
    // h = 257
    for (int i = 0; i < h; i++) {
        float* ptr = (float *)Mat_row(input,i);
        // 取对应的权重
        float a = ((float *)bn->a_data_mat->data)[i];
        float b = ((float *)bn->b_data_mat->data)[i];
        // w = 3
        for (int j = 0; j < w; j++) {
            ptr[j] = b * ptr[j] + a;
        }
    }
}

```

### 2.5 BatchNorm1D NEON优化

BatchNorm1D的优化分为两种情况，一种是可以packing的，也就是`h`维度能被4整除，一种是`h`不能被4整除。

当`h`能被4整除，假设输入数据是被packing过的。

```c
if (input->elempack == 4) {
    if (input->dims == 2) {
        int w = input->w;
        int h = input->h;

        #pragma omp parallel for num_threads(NUM_THREADS)
        // h = 64 (256/4)
        for (int i = 0; i < h; i++) {
            // 权重并不需要做packing，因为权重是纵向分布的，可以画一下数据的内存分布就清楚了
            float32x4_t _a = vld1q_f32((const float*)bn->a_data_mat->data + i * 4);
            float32x4_t _b = vld1q_f32((const float*)bn->b_data_mat->data + i * 4);

            // 取一行数据(4,3)
            float* ptr = (float *)Mat_row(input,i);

            // w = 3
            for (int j = 0; j < w; j++) {
                float32x4_t _p = vld1q_f32(ptr);    // 一次读取4个
                _p = vmlaq_f32(_a, _p, _b);         // 4个一组进行计算
                vst1q_f32(ptr, _p);

                ptr += 4;
            }
        }
    }
}
```

当`h`不能被4整除：

```c
if (input->dims == 2) {
    int w = input->w;
    int h = input->h;

    #pragma omp parallel for num_threads(NUM_THREADS)
    // h = 257
    for (int i = 0; i < h; i++) {
        float* ptr = (float *)Mat_row(input,i);

        float a = ((float *)bn->a_data_mat->data)[i];
        float b = ((float *)bn->b_data_mat->data)[i];

        int j = 0;

        // 拷贝权重为4份
        float32x4_t _a = vdupq_n_f32(a);
        float32x4_t _b = vdupq_n_f32(b);

        // 当w大于4时，可以进行neon操作
        for (; j + 3 < w; j += 4) {
            float32x4_t _p = vld1q_f32(ptr);
            _p = vmlaq_f32(_a, _p, _b);
            vst1q_f32(ptr, _p);

            ptr += 4;
        }
        // 否则进行普通操作
        for (; j < w; j++) {
            *ptr = b * *ptr + a;

            ptr++;
        }
    }
}
```

#### 2.6 运行结果

```c
(1,128)
normal 2~3us 
arm_1  2~3us
```