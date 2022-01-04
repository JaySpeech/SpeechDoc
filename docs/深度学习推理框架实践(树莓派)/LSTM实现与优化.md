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
weight_ih_l0 = np.float32(model.mask.weight_ih_l0.detach().cpu().numpy())
weight_ih_l0 = list(weight_ih_l0.reshape(-1))
print("weight_ih_l0 len:" + str(len(weight_ih_l0)))
data = struct.pack('f'*len(weight_ih_l0),*weight_ih_l0)
with open("mask_param.bin",'ab+') as f:
    f.write(data)

weight_hh_l0 = np.float32(model.mask.weight_hh_l0.detach().cpu().numpy())
weight_hh_l0 = list(weight_hh_l0.reshape(-1))
print("weight_hh_l0 len:" + str(len(weight_hh_l0)))
data = struct.pack('f'*len(weight_hh_l0),*weight_hh_l0)
with open("mask_param.bin",'ab+') as f:
    f.write(data)

# note:bias在这里进行相加
bias_h = model.mask.bias_ih_l0 + model.mask.bias_hh_l0
bias_h = np.float32(bias_h.detach().cpu().numpy())
bias_h = list(bias_h.reshape(-1))
print("bias_hh_l0 len:" + str(len(bias_h)))
data = struct.pack('f'*len(bias_h),*bias_h)
with open("mask_param.bin",'ab+') as f:
    f.write(data)
```

### 2.3 LSTM权重导入

LSTM结构体初始化：

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
    // T 1000
    for (int t = 0; t < T; t++) {
        // 取一个时间片
        const float* x = (float *)Mat_row(input,t);
        // num_output 257，计算I F O G
        for (int q = 0; q < num_output; q++){
            const float* bias_c_I = (float *)Mat_row(lstm->bias_c_mat,0);
            const float* bias_c_F = (float *)Mat_row(lstm->bias_c_mat,1);
            const float* bias_c_O = (float *)Mat_row(lstm->bias_c_mat,2);
            const float* bias_c_G = (float *)Mat_row(lstm->bias_c_mat,3);

            // 保存 num_output 257 4个节点的输出
            float* gates_data = (float *)Mat_row(lstm->gates_mat,q);

            // gate I F O G weight_xc 256 weight_hc 257
            const float* weight_xc_I = (float *)Mat_row(lstm->weight_xc_mat,num_output * 0 + q);
            const float* weight_xc_F = (float *)Mat_row(lstm->weight_xc_mat,num_output * 1 + q);
            const float* weight_xc_O = (float *)Mat_row(lstm->weight_xc_mat,num_output * 2 + q);
            const float* weight_xc_G = (float *)Mat_row(lstm->weight_xc_mat,num_output * 3 + q);

            const float* weight_hc_I = (float *)Mat_row(lstm->weight_hc_mat,num_output * 0 + q);
            const float* weight_hc_F = (float *)Mat_row(lstm->weight_hc_mat,num_output * 1 + q);
            const float* weight_hc_O = (float *)Mat_row(lstm->weight_hc_mat,num_output * 2 + q);
            const float* weight_hc_G = (float *)Mat_row(lstm->weight_hc_mat,num_output * 3 + q);

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

                I += weight_hc_I[i] * h_cont;
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

