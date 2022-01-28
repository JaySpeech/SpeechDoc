# PReLU实现和优化

## 1.PReLU实现

### 1.1 PReLU原理

$$
\operatorname{PReLU}(x)=\left\{\begin{array}{ll}
x, & \text { if } x \geq 0 \\
a x, & \text { otherwise }
\end{array}\right.
$$

`PReLU`的参数可以进行如下设置：

```powershell
Here aaa is a learnable parameter. When called without arguments, nn.PReLU() uses a single parameter aaa across all input channels. 
If called with nn.PReLU(nChannels), a separate aaa is used for each input channel.
```

### 1.2 pytorch PReLU权重导出

[pytorch PReLU官方文档](https://pytorch.org/docs/stable/generated/torch.nn.PReLU.html?highlight=prelu#torch.nn.PReLU)


当使用`torch.nn.PReLU()`不进行任何设置时，参数大小为1。

```python
print(model.mask.repeat0_block0_prelu1.weight.shape)

#torch.Size([1])
```

导出方法为：

```c
mask_weight = np.float32(module_dump.weight.detach().cpu().numpy())
mask_weight = list(mask_weight.reshape(-1))
print("prelu_weight len:" + str(len(mask_weight)))
data = struct.pack('f'*len(mask_weight),*mask_weight)
with open(save_file,'ab+') as f:
    f.write(data)
```

### 1.3 PReLU权重导入

`PReLU`结构体初始化和导入方法比较简单：

```c
PReLU *PReLU_create(int num_slope){
    PReLU* prelu = (PReLU *)malloc(sizeof(PReLU));
    if(prelu == NULL){
        return NULL;
    }

    prelu->num_slope = num_slope;
    prelu->slope_data_mat = Mat_1D_create(num_slope,4u,1);

    return prelu;
}

int PReLU_load_variables(PReLU *prelu, char *file){
    if(prelu == NULL){
        return -1;
    }

    FILE * weight_bin_file = fopen(file,"rb");
    if(weight_bin_file == NULL){
        return -1;
    }

    fread(prelu->slope_data_mat->data, sizeof(float), prelu->num_slope, weight_bin_file);

    fclose(weight_bin_file);

    return 0;
}
```

### 1.4 PReLU实现

```c
if (dims == 2) {
    int w = input->w;
    int h = input->h;

    #pragma omp parallel for num_threads(NUM_THREADS)
    for (int i = 0; i < h; i++) {
        float* ptr = (float *)Mat_row(input,i);
        float slope = prelu->num_slope > 1 ? \
                    ((float *)prelu->slope_data_mat->data)[i] : ((float *)prelu->slope_data_mat->data)[0];

        for (int j = 0; j < w; j++) {
            if (ptr[j] < 0)
                ptr[j] *= slope;
        }
    }
}
```

### 1.5 PReLU NEON优化

PReLU的优化分为两种情况，一种是可以packing的，也就是`h`维度能被4整除，一种是`h`不能被4整除。

当`h`能被4整除，假设输入数据是被packing过的。

if (elempack == 4) {
    float32x4_t _zero = vdupq_n_f32(0.f);

    if (dims == 2) {
        int w = input->w;
        int h = input->h;

        #pragma omp parallel for num_threads(NUM_THREADS)
        for (int i = 0; i < h; i++) {
            float* ptr = (float *)Mat_row(input,i);
            float32x4_t _slope = prelu->num_slope > 1 ? \
                                    vld1q_f32((const float*)prelu->slope_data_mat->data + i * 4) : \
                                    vdupq_n_f32(((float *)prelu->slope_data_mat->data)[0]);

            for (int j = 0; j < w; j++) {
                float32x4_t _p = vld1q_f32(ptr);            // 连续读取4个数据
                uint32x4_t _lemask = vcleq_f32(_p, _zero);  // vcleq_f32 Floating-point compare less than or equal
                float32x4_t _ps = vmulq_f32(_p, _slope);    // 4个连续乘
                _p = vbslq_f32(_lemask, _ps, _p);           // will select first if _lemask 0, second if _lemask 1
                vst1q_f32(ptr, _p);

                ptr += 4;
            }
        }
    }
}

当`h`不能被4整除：

```c
if (input->dims == 2) {
    int w = input->w;
    int h = input->h;

    #pragma omp parallel for num_threads(NUM_THREADS)
    for (int i = 0; i < h; i++) {
        float* ptr = (float *)Mat_row(input,i);

        float a = ((float *)bn->a_data_mat->data)[i];
        float b = ((float *)bn->b_data_mat->data)[i];

        int j = 0;

        // 权重被复制4份
        float32x4_t _a = vdupq_n_f32(a);
        float32x4_t _b = vdupq_n_f32(b);

        // 当 w >= 4，可以进行neon操作
        for (; j + 3 < w; j += 4) {
            float32x4_t _p = vld1q_f32(ptr);
            _p = vmlaq_f32(_a, _p, _b);
            vst1q_f32(ptr, _p);

            ptr += 4;
        }
        for (; j < w; j++) {
            *ptr = b * *ptr + a;

            ptr++;
        }
    }
}
```

#### 1.6 运行结果

```c
(1,128)
normal 2~3us 
arm_1  2~3us
```


