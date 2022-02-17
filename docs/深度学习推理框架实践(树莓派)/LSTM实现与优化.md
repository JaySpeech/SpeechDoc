# LSTM实现与优化

## 1.基于LSTM的语音增强网络实现

### 1.1 IRM语音增强网络

设计一个简单的IRM语音增强网络。

```python
self.mask = torch.nn.LSTM(256, 257, num_layers=1, bidirectional=False, batch_first=True)
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

```powershell
pip install onnx-simplifier
python -m onnxsim ncnn_test.onnx ncnn_test_op.onnx 
```

简化后的网络结构。

![](img/LSTM实现与优化/lstm_onnx_op.png ':size=25%')


### 1.2 NCNN中运行

将模型的onnx格式转成ncnn格式。

param文件内容如下，用另外一个形式表述了网络架构。

```powershell
7767517
3 5
Input            x                        0 1 x
LSTM             LSTM_10                  1 3 x 48 45 46 0=257 1=263168 2=0
Sigmoid          Sigmoid_13               1 1 48 y
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

* 忘记门，决定我们会从细胞状态$C_{t-1}$中丢弃什么信息。来自先前隐藏状态$h_{t-1}$和来自当前输入$x_{t}$的信息通过`sigmoid`函数传递。值介于0和1之间。越接近0越容易遗忘，越接近1则意味着要保留。($C_{t-1}$*忘记门来决定丢弃的程度)

![](img/LSTM实现与优化/lstm_01.png ':size=50%')

* 更新门，确定什么样的新信息被存放在细胞状态中。

    这里包含两个部分。首先，`sigmoid`层称 “输入门层” 决定什么值我们将要更新。
    然后，一个 tanh 层创建一个新的候选值向量，$\tilde{C}_t$，会被加入到状态中。下一步，我们会将这两个信息来产生对状态的更新。

![](img/LSTM实现与优化/lstm_02.png ':size=50%')

* 更新细胞状态

    我们把旧状态与$f_t$相乘，丢弃掉我们确定需要丢弃的信息。接着加上$i_t * \tilde{C}_t$。这就是新的候选值，根据我们决定更新每个状态的程度进行变化。

![](img/LSTM实现与优化/lstm_03.png ':size=50%')

* 输出门

    最后，根据当前细胞状态$C_{t}$、先前隐藏状态$h_{t-1}$和来自当前输入$x_{t}$最终确定输出值。

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

对照LSTM公式，先来打印一下LSTM的参数大小。

```python
print(model.mask.weight_ih_l0.shape)
print(model.mask.weight_hh_l0.shape)
print(model.mask.bias_ih_l0.shape)
print(model.mask.bias_hh_l0.shape)

#torch.Size([1028, 256])
#torch.Size([1028, 257])
#torch.Size([1028])  4*257
#torch.Size([1028])  4*257
```

权重导出为二进制文件。

```python
def lstm_dump(module_dump:torch.nn.Module,save_file:String):
    if os.path.exists(save_file):
        os.remove(save_file)

    print("------dump "+ save_file+'------')

    weight_ih_l0 = np.float32(module_dump.weight_ih_l0.detach().cpu().numpy())
    weight_ih_l0 = list(weight_ih_l0.reshape(-1))
    print("weight_ih_l0 len:" + str(len(weight_ih_l0)))
    data = struct.pack('f'*len(weight_ih_l0),*weight_ih_l0)
    with open(save_file,'ab+') as f:
        f.write(data)

    weight_hh_l0 = np.float32(module_dump.weight_hh_l0.detach().cpu().numpy())
    weight_hh_l0 = list(weight_hh_l0.reshape(-1))
    print("weight_hh_l0 len:" + str(len(weight_hh_l0)))
    data = struct.pack('f'*len(weight_hh_l0),*weight_hh_l0)
    with open(save_file,'ab+') as f:
        f.write(data)

    bias_h = module_dump.bias_ih_l0 + module_dump.bias_hh_l0
    bias_h = np.float32(bias_h.detach().cpu().numpy())
    bias_h = list(bias_h.reshape(-1))
    print("bias_hh_l0 len:" + str(len(bias_h)))
    data = struct.pack('f'*len(bias_h),*bias_h)
    with open(save_file,'ab+') as f:
        f.write(data)
```

### 2.3 LSTM权重导入

`LSTM`结构体初始化：

```c
LSTM *LSTM_create(int input_size, int hidden_size, bool bias_used){
    LSTM* lstm = (LSTM *)malloc(sizeof(LSTM));
    if(lstm == NULL){
        return NULL;
    }

    lstm->input_size = input_size;
    lstm->hidden_size = hidden_size;
    lstm->bias_used = bias_used;

    lstm->weight_xc_mat = Mat_2D_create(input_size,4*hidden_size,4,1);
    lstm->weight_hc_mat = Mat_2D_create(hidden_size,4*hidden_size,4,1);
    lstm->bias_c_mat = Mat_2D_create(hidden_size,4,4,1);

    lstm->hidden_mat = Mat_1D_create(hidden_size,4,1);
    lstm->cell_mat = Mat_1D_create(hidden_size,4,1);

    float *hidden_ptr = (float *)lstm->hidden_mat->data;
    float *cell_ptr = (float *)lstm->cell_mat->data;
    for(int i = 0; i < hidden_size; i++){
        hidden_ptr[i] = 0.0f;
        cell_ptr[i] = 0.0f;
    }

    lstm->gates_mat = Mat_2D_create(4,hidden_size,4,1);

    lstm->out_mat = NULL;

    return lstm;
}
```

权重导入：

```c
int LSTM_load_variables(LSTM *lstm, char* file){
    if(lstm == NULL){
        return -1;
    }

    FILE * weight_bin_file = fopen(file,"rb");
    if(weight_bin_file == NULL){
        return -1;
    }

    fread(lstm->weight_xc_mat->data,sizeof(float),lstm->input_size*lstm->hidden_size*4,weight_bin_file);
    //Mat_2D_float_printf(lstm->weight_xc_mat);
    
    fread(lstm->weight_hc_mat->data,sizeof(float),lstm->hidden_size*lstm->hidden_size*4,weight_bin_file);
    //Mat_2D_float_printf(lstm->weight_hc_mat);

    if(lstm->bias_used == true){
        fread(lstm->bias_c_mat->data,sizeof(float),lstm->hidden_size*4,weight_bin_file);
        //Mat_2D_float_printf(lstm->bias_c_mat);
    }

    fclose(weight_bin_file);

    return 0;
}
```

### 2.4 LSTM实现

这里假设输入为`(1000,256)`，输出为`(1000,257)`。

```c
int LSTM_process(LSTM *lstm, Mat *input){

    if(lstm->out_mat == NULL){
        lstm->out_mat = Mat_2D_create(lstm->hidden_size,input->h,sizeof(float),1);
    }

    int T = input->h;
    int num_output = lstm->hidden_size;
    float *hidden_state = (float *)lstm->hidden_mat->data;
    float *cell_state = (float *)lstm->cell_mat->data;

    // 计算矩阵乘加
    // T 1000 由于LSTM的当前输出依赖上一次的状态，无法进行并发
    for (int t = 0; t < T; t++) {
        // 取一个时间片
        const float* x = (float *)Mat_row(input,t);
        // num_output 257，计算I F O G
        for (int q = 0; q < num_output; q++){
            const float* bias_c_I = (float *)Mat_row(lstm->bias_c_mat,0);
            const float* bias_c_F = (float *)Mat_row(lstm->bias_c_mat,1);
            const float* bias_c_O = (float *)Mat_row(lstm->bias_c_mat,3);
            const float* bias_c_G = (float *)Mat_row(lstm->bias_c_mat,2);

            // 保存 num_output 257 4个节点的输出
            float* gates_data = (float *)Mat_row(lstm->gates_mat,q);

            // gate I F O G weight_xc 256 weight_hc 257
            const float* weight_xc_I = (float *)Mat_row(lstm->weight_xc_mat,num_output * 0 + q);
            const float* weight_xc_F = (float *)Mat_row(lstm->weight_xc_mat,num_output * 1 + q);
            const float* weight_xc_O = (float *)Mat_row(lstm->weight_xc_mat,num_output * 3 + q);
            const float* weight_xc_G = (float *)Mat_row(lstm->weight_xc_mat,num_output * 2 + q);

            const float* weight_hc_I = (float *)Mat_row(lstm->weight_hc_mat,num_output * 0 + q);
            const float* weight_hc_F = (float *)Mat_row(lstm->weight_hc_mat,num_output * 1 + q);
            const float* weight_hc_O = (float *)Mat_row(lstm->weight_hc_mat,num_output * 3 + q);
            const float* weight_hc_G = (float *)Mat_row(lstm->weight_hc_mat,num_output * 2 + q);

            float I = bias_c_I[q];
            float F = bias_c_F[q];
            float O = bias_c_O[q];
            float G = bias_c_G[q];

            // 256*256
            for (int i = 0; i < lstm->input_size; i++){
                float xi = x[i];

                I += weight_xc_I[i] * xi;
                F += weight_xc_F[i] * xi;
                O += weight_xc_O[i] * xi;
                G += weight_xc_G[i] * xi;
            }

            // 257*257
            for (int i = 0; i < num_output; i++){
                float h_cont = hidden_state[i];

                I += weight_hc_I[i] * h_cont;  // 无法使用openmp进行优化，多线程不能同时操作I
                F += weight_hc_F[i] * h_cont;
                O += weight_hc_O[i] * h_cont;
                G += weight_hc_G[i] * h_cont;
            }

            gates_data[0] = I;
            gates_data[1] = F;
            gates_data[2] = O;
            gates_data[3] = G;
        }


        // lstm最后输出
        float* output_data = (float *)Mat_row(lstm->out_mat,t);
        for (int q = 0; q < num_output; q++) {
            const float* gates_data = (float *)Mat_row(lstm->gates_mat,q);

            float I = gates_data[0];
            float F = gates_data[1];
            float O = gates_data[2];
            float G = gates_data[3];

            I = 1.f / (1.f + exp(-I));
            F = 1.f / (1.f + exp(-F));
            O = 1.f / (1.f + exp(-O));
            G = tanh(G);

            float cell2 = F * cell_state[q] + I * G;
            float H = O * tanh(cell2);
            cell_state[q] = cell2;
            hidden_state[q] = H;
            output_data[q] = H;
        }
    }

    return 0;
}
```

### 2.5 LSTM NEON优化

#### 2.5.1 LSTM权重Pack

LSTM中有IFOG四个计算单元，正好符合并行的要求。

```c
Mat *weight_xc_mat = Mat_2D_create(lstm->input_size,4*lstm->hidden_size,4,1);
Mat *weight_hc_mat = Mat_2D_create(lstm->hidden_size,4*lstm->hidden_size,4,1);
Mat *bias_c_mat = Mat_2D_create(lstm->hidden_size,4,4,1);

// 创建pack权重mat
lstm->weight_xc_packed_mat = Mat_2D_create(input_size,hidden_size,16u,4);
lstm->weight_hc_packed_mat = Mat_2D_create(hidden_size,hidden_size,16u,4);
lstm->bias_c_packed_mat = Mat_2D_create(hidden_size,1,16u,4);

// pack IFOG
const float* bias_c_I = (float *)Mat_row(bias_c_mat,0);
const float* bias_c_F = (float *)Mat_row(bias_c_mat,1);
const float* bias_c_O = (float *)Mat_row(bias_c_mat,3);
const float* bias_c_G = (float *)Mat_row(bias_c_mat,2);

float* bias_c_IFOG = (float *)Mat_row(lstm->bias_c_packed_mat,0);

int num_output = lstm->hidden_size;

// 每次分别从IFOG权重中各取一个值进行Pack
for (int q = 0; q < num_output; q++) {
    bias_c_IFOG[0] = bias_c_I[q];
    bias_c_IFOG[1] = bias_c_F[q];
    bias_c_IFOG[2] = bias_c_O[q];
    bias_c_IFOG[3] = bias_c_G[q];

    bias_c_IFOG += 4;

    const float* weight_xc_I = (float *)Mat_row(weight_xc_mat,num_output * 0 + q);
    const float* weight_xc_F = (float *)Mat_row(weight_xc_mat,num_output * 1 + q);
    const float* weight_xc_O = (float *)Mat_row(weight_xc_mat,num_output * 3 + q);
    const float* weight_xc_G = (float *)Mat_row(weight_xc_mat,num_output * 2 + q);

    const float* weight_hc_I = (float *)Mat_row(weight_hc_mat,num_output * 0 + q);
    const float* weight_hc_F = (float *)Mat_row(weight_hc_mat,num_output * 1 + q);
    const float* weight_hc_O = (float *)Mat_row(weight_hc_mat,num_output * 3 + q);
    const float* weight_hc_G = (float *)Mat_row(weight_hc_mat,num_output * 2 + q);

    float* weight_xc_IFOG = (float *)Mat_row(lstm->weight_xc_packed_mat,q);
    float* weight_hc_IFOG = (float *)Mat_row(lstm->weight_hc_packed_mat,q);

    for (int i = 0; i < lstm->input_size; i++) {
        weight_xc_IFOG[0] = weight_xc_I[i];
        weight_xc_IFOG[1] = weight_xc_F[i];
        weight_xc_IFOG[2] = weight_xc_O[i];
        weight_xc_IFOG[3] = weight_xc_G[i];

        weight_xc_IFOG += 4;
    }

    for (int i = 0; i < num_output; i++) {
        weight_hc_IFOG[0] = weight_hc_I[i];
        weight_hc_IFOG[1] = weight_hc_F[i];
        weight_hc_IFOG[2] = weight_hc_O[i];
        weight_hc_IFOG[3] = weight_hc_G[i];

        weight_hc_IFOG += 4;
    }
}
```

#### 2.5.2 IFOG计算优化

`vmlaq_lane_f32`函数的定义：

```c
// Vector multiply accumulate with scalar
vmlaq_lane_f32 (float32x4_t a, float32x4_t b, float32x2_t v, const int lane)
RESULT[I] = a[i] + (b[i] * v[lane]) for i = 0 to 3
lane minimum: 0; maximum: 1
```

```c
// 取一个时间片数据
const float* x = (float *)Mat_row(input, t);
for (int q = 0; q < num_output; q++) {
    const float* bias_c_IFOG = (const float*)lstm->bias_c_packed_mat->data + q * 4;

    // gate I F O G
    const float* weight_xc_IFOG = (float *)Mat_row(lstm->weight_xc_packed_mat,q);

    const float* weight_hc_IFOG = (float *)Mat_row(lstm->weight_hc_packed_mat,q);

    float32x4_t _IFOG = vld1q_f32(bias_c_IFOG); // load 4个 bias float
    float32x4_t _sum1 = vdupq_n_f32(0.f);
    float32x4_t _sum2 = vdupq_n_f32(0.f);
    float32x4_t _sum3 = vdupq_n_f32(0.f);

    int i = 0;
    // 256*256计算，一次计算4个
    for (; i + 3 < lstm->input_size; i += 4){
        float32x4_t _xi = vld1q_f32(x + i); // load x+i 输入数据连续4个float

        // 读取4*4个权重数据，分别对应4个输入
        float32x4_t _weight_xc_IFOG_0 = vld1q_f32(weight_xc_IFOG);
        float32x4_t _weight_xc_IFOG_1 = vld1q_f32(weight_xc_IFOG + 4);
        float32x4_t _weight_xc_IFOG_2 = vld1q_f32(weight_xc_IFOG + 8);
        float32x4_t _weight_xc_IFOG_3 = vld1q_f32(weight_xc_IFOG + 12);

        // 每个input分别和4个权重进行乘加，这里有4个input
        _IFOG = vmlaq_lane_f32(_IFOG, _weight_xc_IFOG_0, vget_low_f32(_xi), 0);
        _sum1 = vmlaq_lane_f32(_sum1, _weight_xc_IFOG_1, vget_low_f32(_xi), 1);
        _sum2 = vmlaq_lane_f32(_sum2, _weight_xc_IFOG_2, vget_high_f32(_xi), 0);
        _sum3 = vmlaq_lane_f32(_sum3, _weight_xc_IFOG_3, vget_high_f32(_xi), 1);

        weight_xc_IFOG += 16;
    }

    // 不够4整除的数据，使用vmlaq_f32进行input和4个权重乘加操作
    for (; i < lstm->input_size; i++) {
        float xi = x[i];

        float32x4_t _xi = vdupq_n_f32(xi);
        float32x4_t _weight_xc_IFOG = vld1q_f32(weight_xc_IFOG);
        _IFOG = vmlaq_f32(_IFOG, _weight_xc_IFOG, _xi);

        weight_xc_IFOG += 4;
    }

    i = 0;

    // 和前面的操作类似
    for (; i + 3 < num_output; i += 4) {
        float32x4_t _h_cont = vld1q_f32((const float*)hidden_state + i);

        float32x4_t _weight_hc_IFOG_0 = vld1q_f32(weight_hc_IFOG);
        float32x4_t _weight_hc_IFOG_1 = vld1q_f32(weight_hc_IFOG + 4);
        float32x4_t _weight_hc_IFOG_2 = vld1q_f32(weight_hc_IFOG + 8);
        float32x4_t _weight_hc_IFOG_3 = vld1q_f32(weight_hc_IFOG + 12);

        _IFOG = vmlaq_lane_f32(_IFOG, _weight_hc_IFOG_0, vget_low_f32(_h_cont), 0);
        _sum1 = vmlaq_lane_f32(_sum1, _weight_hc_IFOG_1, vget_low_f32(_h_cont), 1);
        _sum2 = vmlaq_lane_f32(_sum2, _weight_hc_IFOG_2, vget_high_f32(_h_cont), 0);
        _sum3 = vmlaq_lane_f32(_sum3, _weight_hc_IFOG_3, vget_high_f32(_h_cont), 1);

        weight_hc_IFOG += 16;
    }

    for (; i < num_output; i++) {
        float h_cont = hidden_state[i];

        float32x4_t _h_cont = vdupq_n_f32(h_cont);
        float32x4_t _weight_hc_IFOG = vld1q_f32(weight_hc_IFOG);
        _IFOG = vmlaq_f32(_IFOG, _weight_hc_IFOG, _h_cont);

        weight_hc_IFOG += 4;
    }

    float* gates_data = (float *)Mat_row(lstm->gates_mat,q);

    // _IFOG _sum1 _sum2 _sum3 累加
    _IFOG = vaddq_f32(_IFOG, _sum1);
    _sum2 = vaddq_f32(_sum2, _sum3);
    _IFOG = vaddq_f32(_IFOG, _sum2);

    vst1q_f32(gates_data, _IFOG);
}

```

#### 2.5.2 最终输出计算优化

```c
float* output_data = (float *)Mat_row(lstm->out_mat,t);

float* cell_ptr = cell_state;
float* hidden_ptr = hidden_state;

int q = 0;

// gates_data (257,4)
for (; q + 3 < num_output; q += 4) {
    const float* gates_data = (float *)Mat_row(lstm->gates_mat,q);

    float32x4x4_t _IFOG_4x4 = vld4q_f32(gates_data); // 读取16个

    float32x4_t _I = sigmoid_ps(_IFOG_4x4.val[0]);
    float32x4_t _F = sigmoid_ps(_IFOG_4x4.val[1]);
    float32x4_t _O = sigmoid_ps(_IFOG_4x4.val[2]);
    float32x4_t _G = tanh_ps(_IFOG_4x4.val[3]);

    float32x4_t _cell2 = vaddq_f32(vmulq_f32(_F, vld1q_f32(cell_ptr)), vmulq_f32(_I, _G));
    float32x4_t _H = vmulq_f32(_O, tanh_ps(_cell2));

    vst1q_f32(cell_ptr, _cell2);
    vst1q_f32(hidden_ptr, _H);
    vst1q_f32(output_data, _H);

    cell_ptr += 4;
    hidden_ptr += 4;
    output_data += 4;
}

for (; q < num_output; q++) {
    const float* gates_data = (float *)Mat_row(lstm->gates_mat,q);

    float I = gates_data[0];
    float F = gates_data[1];
    float O = gates_data[2];
    float G = gates_data[3];

    I = 1.f / (1.f + exp(-I));
    F = 1.f / (1.f + exp(-F));
    O = 1.f / (1.f + exp(-O));
    G = tanh(G);

    float cell2 = F * *cell_ptr + I * G;
    float H = O * tanh(cell2);

    *cell_ptr++ = cell2;
    *hidden_ptr++ = H;
    *output_data++ = H;
}
```


#### 2.5.3 运行结果

LSTM内部存在3个`sigmoid`和2个`tanh`计算，运算比较费时，并不能很好的利用多核优势。

```c
LSTM input 256 hidden 257
normal 1239us
arm_1 540us
无多核优化
```