# GRU实现与优化

## 1.基于GRU的语音增强网络实现

### 1.1 IRM语音增强网络

设计一个简单的IRM语音增强网络。

```python
self.mask = torch.nn.GRU(256, 257, num_layers=1, bidirectional=False, batch_first=True)
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

![](img/GRU实现与优化/gru_onnx.png ':size=25%')

使用`onnx-simplifier`进行简化。

```powershell
pip install onnx-simplifier
python -m onnxsim ncnn_test.onnx ncnn_test_op.onnx 
```

简化后的网络结构。

![](img/GRU实现与优化/gru_onnx_op.png ':size=25%')


### 1.2 NCNN中运行

将模型的onnx格式转成ncnn格式。

param文件内容如下，用另外一个形式表述了网络架构。

```powershell
7767517
3 4
Input            x                        0 1 x
GRU              GRU_6                    1 2 x 38 37 0=257 1=197376 2=0
Sigmoid          Sigmoid_8                1 1 38 y
```
![](img/GRU实现与优化/gru_ncnn.png ':size=25%')


## 2.GRU实现

### 2.1 GRU基本原理

GRU是LSTM网络的一种效果很好的变体，没有了细胞状态，较LSTM网络的结构更加简单。

![](img/GRU实现与优化/gru_计算公式.png ':size=50%')

* 重置门$r_{t}$

    重置门根据上一时刻输出$h_{t-1}$和当前时刻输入$x_{t}$确定上一时刻的输出$h_{t-1}$需要被遗忘多多少。重置门越小，前一状态的信息被写入的越少。


* 更新门$z_{t}$

    更新门用于控制前一时刻的状态信息被带入到当前状态中的程度，更新门的值越大说明前一时刻的状态信息带入越多。


### 2.2 pytorch GRU和权重导出

[pytorch GRU官方文档](https://pytorch.org/docs/stable/generated/torch.nn.GRU.html)

从GRU官方文档可以知道，LSTM的参数为`weight_ih_l[k]`,`weight_hh_l[k]`,`bias_ih_l[k]`,`bias_hh_l[k]`。

$$
\begin{aligned}
r_{t} &=\sigma\left(W_{i r} x_{t}+b_{i r}+W_{h r} h_{(t-1)}+b_{h r}\right) \\
z_{t} &=\sigma\left(W_{i z} x_{t}+b_{i z}+W_{h z} h_{(t-1)}+b_{h z}\right) \\
n_{t} &=\tanh \left(W_{i n} x_{t}+b_{i n}+r_{t} *\left(W_{h n} h_{(t-1)}+b_{h n}\right)\right) \\
h_{t} &=\left(1-z_{t}\right) * n_{t}+z_{t} * h_{(t-1)}
\end{aligned}
$$

对照GRU公式，先来打印一下GRU的参数大小。

```python
print(model.mask.weight_ih_l0.shape)
print(model.mask.weight_hh_l0.shape)
print(model.mask.bias_ih_l0.shape)
print(model.mask.bias_hh_l0.shape)

#torch.Size([771, 256])
#torch.Size([771, 257])
#torch.Size([771])  3*257
#torch.Size([771])  3*257
```

权重导出为二进制文件。

```python
def gru_dump(module_dump:torch.nn.Module,save_file:String):
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

    hidden_size = int(module_dump.bias_ih_l0.shape[0]/3)

    bias_h_0 = module_dump.bias_ih_l0[0:hidden_size*2] + module_dump.bias_hh_l0[0:hidden_size*2]
    bias_h_0 = np.float32(bias_h_0.detach().cpu().numpy())
    bias_h_0 = list(bias_h_0.reshape(-1))
    print("bias_h_0 len:" + str(len(bias_h_0)))
    data = struct.pack('f'*len(bias_h_0),*bias_h_0)
    with open(save_file,'ab+') as f:
        f.write(data)

    bias_h_1 = module_dump.bias_ih_l0[hidden_size*2:]
    bias_h_1 = np.float32(bias_h_1.detach().cpu().numpy())
    bias_h_1 = list(bias_h_1.reshape(-1))
    print("bias_h_1 len:" + str(len(bias_h_1)))
    data = struct.pack('f'*len(bias_h_1),*bias_h_1)
    with open(save_file,'ab+') as f:
        f.write(data)

    bias_h_2 = module_dump.bias_hh_l0[hidden_size*2:]
    bias_h_2 = np.float32(bias_h_2.detach().cpu().numpy())
    bias_h_2 = list(bias_h_2.reshape(-1))
    print("bias_h_2 len:" + str(len(bias_h_2)))
    data = struct.pack('f'*len(bias_h_2),*bias_h_2)
    with open(save_file,'ab+') as f:
        f.write(data)
```

### 2.3 GRU权重导入

`GRU`结构体初始化：

```c
GRU *GRU_create(int input_size, int hidden_size, bool bias_used){
    GRU* gru = (GRU *)malloc(sizeof(GRU));
    if(gru == NULL){
        return NULL;
    }

    gru->input_size = input_size;
    gru->hidden_size = hidden_size;
    gru->bias_used = bias_used;

    gru->weight_xc_mat = Mat_2D_create(input_size,3*hidden_size,4,1);
    gru->weight_hc_mat = Mat_2D_create(hidden_size,3*hidden_size,4,1);
    gru->bias_c_mat = Mat_2D_create(hidden_size,4,4,1);

    gru->hidden_mat = Mat_1D_create(hidden_size,4,1);

    float *hidden_ptr = (float *)gru->hidden_mat->data;
    for(int i = 0; i < hidden_size; i++){
        hidden_ptr[i] = 0.0f;
    }

    gru->gates_mat = Mat_2D_create(2,hidden_size,4,1);

    gru->out_mat = NULL;

    return gru;
}
```

权重导入：

```c
int GRU_load_variables(GRU *gru, char* file){
    if(gru == NULL){
        return -1;
    }

    FILE * weight_bin_file = fopen(file,"rb");
    if(weight_bin_file == NULL){
        return -1;
    }

    size_t ret;
    ret = fread(gru->weight_xc_mat->data,sizeof(float),gru->input_size*gru->hidden_size*3,weight_bin_file);
    if(ret == 0 || ret == -1){
        return -1;
    }    
    //Mat_2D_float_printf(gru->weight_xc_mat);
    
    ret = fread(gru->weight_hc_mat->data,sizeof(float),gru->hidden_size*gru->hidden_size*3,weight_bin_file);
    if(ret == 0 || ret == -1){
        return -1;
    }    
    //Mat_2D_float_printf(gru->weight_hc_mat);

    if(gru->bias_used == true){
        ret = fread(gru->bias_c_mat->data,sizeof(float),gru->hidden_size*4,weight_bin_file);
        if(ret == 0 || ret == -1){
            return -1;
        }
        //Mat_2D_float_printf(gru->bias_c_mat);
    }

    fclose(weight_bin_file);

    return 0;
}
```

### 2.4 GRU实现

实现过程和LSTM类似，对照2.2中的公式，代码理解不难。

```c
int GRU_process(GRU *gru, Mat *input){
    if(gru == NULL){
        return -1;
    }

    if(gru->out_mat == NULL){
        gru->out_mat = Mat_2D_create(gru->hidden_size,input->h,sizeof(float),1);
    }

    int T = input->h;
    int num_output = gru->hidden_size;
    float *hidden_state = (float *)gru->hidden_mat->data;

    for (int t = 0; t < T; t++) {
        const float* x = (float *)Mat_row(input,t);

        for (int q = 0; q < num_output; q++) {
            float* gates_data = (float *)Mat_row(gru->gates_mat,q);

            // gate reset update
            const float* bias_c_R = (float *)Mat_row(gru->bias_c_mat,0);
            const float* bias_c_U = (float *)Mat_row(gru->bias_c_mat,1);

            const float* weight_xc_R = (float *)Mat_row(gru->weight_xc_mat,num_output * 0 + q);
            const float* weight_xc_U = (float *)Mat_row(gru->weight_xc_mat,num_output * 1 + q);
            const float* weight_hc_R = (float *)Mat_row(gru->weight_hc_mat,num_output * 0 + q);
            const float* weight_hc_U = (float *)Mat_row(gru->weight_hc_mat,num_output * 1 + q);

            float R = bias_c_R[q];
            float U = bias_c_U[q];

            for (int i = 0; i < gru->input_size; i++) {
                float xi = x[i];

                R += weight_xc_R[i] * xi;
                U += weight_xc_U[i] * xi;
            }

            for (int i = 0; i < num_output; i++) {
                float h_cont = hidden_state[i];

                R += weight_hc_R[i] * h_cont;
                U += weight_hc_U[i] * h_cont;
            }

            // sigmoid(R)
            // sigmoid(U)
            R = 1.f / (1.f + exp(-R));
            U = 1.f / (1.f + exp(-U));

            // gate new
            const float* bias_c_WN = (float *)Mat_row(gru->bias_c_mat,2);
            const float* bias_c_BN = (float *)Mat_row(gru->bias_c_mat,3);

            const float* weight_xc_N =  (float *)Mat_row(gru->weight_xc_mat,num_output * 2 + q);
            const float* weight_hc_N =  (float *)Mat_row(gru->weight_hc_mat,num_output * 2 + q);

            float N = bias_c_BN[q];

            for (int i = 0; i < num_output; i++) {
                float h_cont = hidden_state[i];

                N += weight_hc_N[i] * h_cont;
            }

            N = bias_c_WN[q] + R * N;

            for (int i = 0; i < gru->input_size; i++) {
                float xi = x[i];

                N += weight_xc_N[i] * xi;
            }

            // tanh(N)
            N = tanh(N);

            gates_data[0] = U;
            gates_data[1] = N;
        }

        // h_t := (1 - update) .* new + update .* h_{t-1}
        float* output_data = (float *)Mat_row(gru->out_mat,t);
        for (int q = 0; q < num_output; q++) {
            const float* gates_data = (float *)Mat_row(gru->gates_mat,q);

            float U = gates_data[0];
            float N = gates_data[1];

            float H = (1 - U) * N + U * hidden_state[q];

            hidden_state[q] = H;
            output_data[q] = H;
            //printf("%f ",H);
        }
        //printf("\n");
        //getchar();
    }

    return 0;
}
```

### 2.5 GRU NEON优化

假设输入为`(1,256)`，输出`(1,257)`,和输入相关的其中一个全连接层(共有3个)权重大小为`(256,257)`，
那么一次全连接层操作需要进行`256(内循环)*257(外循环)`次乘法操作。NEON优化的思路为先将外循环分为`4*64+1`次，
能被4整除的外循环，一次性执行4次乘法操作。进一步内循环也可以分为`4*64`次，一次读取4个输入，加速读取，并进行4次neon乘加运算。

```powershell
(1,256) (256,257) -> (1,257)

未优化：
外循环输出(257)
    内循环输入(256)
        每次进行一次乘加运算，输入*权重

外循环优化：
外循环输出(64)
    内循环输入(256)
        每次进行一次neon乘加运算，1个输入*4个权重

外循环和内循环优化：
    外循环输出(64)
        内循环输入(64)
            一次性读取4个输入，每次进行4次neon成加运算(1个输入*4个权重)

```

#### 2.5.1 权重预处理

为了符合neon优化内存4float连续的要求，需要将权重重新排列，将4个外循环权重连续存放。

```c
// 这里未不能被4整除的权重多创建了一些空间，浪费掉了
gru->weight_xc_packed_mat = Mat_2D_create(input_size*12,  hidden_size/4 + hidden_size%4, 4, 1);
gru->weight_hc_packed_mat = Mat_2D_create(hidden_size*12 ,hidden_size/4 + hidden_size%4, 4, 1);
gru->bias_c_packed_mat = Mat_2D_create(hidden_size, 1 , 16u ,4);

gru->gates_mat = Mat_2D_create(4*2, hidden_size/4 + hidden_size % 4, 4u, 1);

const float* bias_c_R = (float *)Mat_row(bias_c_mat,0);
const float* bias_c_U = (float *)Mat_row(bias_c_mat,1);
const float* bias_c_WN = (float *)Mat_row(bias_c_mat,2);
const float* bias_c_BN = (float *)Mat_row(bias_c_mat,3);

float* bias_c_RUBNWN = (float *)Mat_row(gru->bias_c_packed_mat,0);

int q = 0;
int num_output = gru->hidden_size;

// 输出能被4整除 每个全连接层包含256个参数，共有257组
// 将257组分为能被4整除部分和不能被4整除部分
for(; q + 3 < num_output; q += 4) {
    // 每个输出对应一个bias 取4个bias组成连续内存，方便neon操作
    bias_c_RUBNWN[0] = bias_c_R[q];
    bias_c_RUBNWN[1] = bias_c_R[q + 1];
    bias_c_RUBNWN[2] = bias_c_R[q + 2];
    bias_c_RUBNWN[3] = bias_c_R[q + 3];

    bias_c_RUBNWN[4] = bias_c_U[q];
    bias_c_RUBNWN[5] = bias_c_U[q + 1];
    bias_c_RUBNWN[6] = bias_c_U[q + 2];
    bias_c_RUBNWN[7] = bias_c_U[q + 3];

    bias_c_RUBNWN[8] = bias_c_BN[q];
    bias_c_RUBNWN[9] = bias_c_BN[q + 1];
    bias_c_RUBNWN[10] = bias_c_BN[q + 2];
    bias_c_RUBNWN[11] = bias_c_BN[q + 3];

    bias_c_RUBNWN[12] = bias_c_WN[q];
    bias_c_RUBNWN[13] = bias_c_WN[q + 1];
    bias_c_RUBNWN[14] = bias_c_WN[q + 2];
    bias_c_RUBNWN[15] = bias_c_WN[q + 3];

    bias_c_RUBNWN += 16;

    const float* weight_xc_R = (float *)Mat_row(weight_xc_mat, num_output * 0 + q);
    const float* weight_xc_U = (float *)Mat_row(weight_xc_mat, num_output * 1 + q);
    const float* weight_xc_N = (float *)Mat_row(weight_xc_mat, num_output * 2 + q);

    const float* weight_xc_R_1 = (float *)Mat_row(weight_xc_mat, num_output * 0 + q + 1);
    const float* weight_xc_U_1 = (float *)Mat_row(weight_xc_mat, num_output * 1 + q + 1);
    const float* weight_xc_N_1 = (float *)Mat_row(weight_xc_mat, num_output * 2 + q + 1);

    const float* weight_xc_R_2 = (float *)Mat_row(weight_xc_mat, num_output * 0 + q + 2);
    const float* weight_xc_U_2 = (float *)Mat_row(weight_xc_mat, num_output * 1 + q + 2);
    const float* weight_xc_N_2 = (float *)Mat_row(weight_xc_mat, num_output * 2 + q + 2);

    const float* weight_xc_R_3 = (float *)Mat_row(weight_xc_mat, num_output * 0 + q + 3);
    const float* weight_xc_U_3 = (float *)Mat_row(weight_xc_mat, num_output * 1 + q + 3);
    const float* weight_xc_N_3 = (float *)Mat_row(weight_xc_mat, num_output * 2 + q + 3);

    const float* weight_hc_R = (float *)Mat_row(weight_hc_mat, num_output * 0 + q);
    const float* weight_hc_U = (float *)Mat_row(weight_hc_mat, num_output * 1 + q);
    const float* weight_hc_N = (float *)Mat_row(weight_hc_mat, num_output * 2 + q);

    const float* weight_hc_R_1 = (float *)Mat_row(weight_hc_mat, num_output * 0 + q + 1);
    const float* weight_hc_U_1 = (float *)Mat_row(weight_hc_mat, num_output * 1 + q + 1);
    const float* weight_hc_N_1 = (float *)Mat_row(weight_hc_mat, num_output * 2 + q + 1);

    const float* weight_hc_R_2 = (float *)Mat_row(weight_hc_mat, num_output * 0 + q + 2);
    const float* weight_hc_U_2 = (float *)Mat_row(weight_hc_mat, num_output * 1 + q + 2);
    const float* weight_hc_N_2 = (float *)Mat_row(weight_hc_mat, num_output * 2 + q + 2);

    const float* weight_hc_R_3 = (float *)Mat_row(weight_hc_mat, num_output * 0 + q + 3);
    const float* weight_hc_U_3 = (float *)Mat_row(weight_hc_mat, num_output * 1 + q + 3);
    const float* weight_hc_N_3 = (float *)Mat_row(weight_hc_mat, num_output * 2 + q + 3);

    float* weight_xc_RUN = (float *)Mat_row(gru->weight_xc_packed_mat, q/4);
    float* weight_hc_RUN = (float *)Mat_row(gru->weight_hc_packed_mat, q/4);

    // 将4个输出 对应的权重参数(256)各取出一个来，组成连续的内存，方便neon操作
    // 先存储RU重置门和更新门相关权重
    for (int i = 0; i < gru->input_size; i++){
        weight_xc_RUN[0] = weight_xc_R[i];
        weight_xc_RUN[1] = weight_xc_R_1[i];
        weight_xc_RUN[2] = weight_xc_R_2[i];
        weight_xc_RUN[3] = weight_xc_R_3[i];

        weight_xc_RUN[4] = weight_xc_U[i];
        weight_xc_RUN[5] = weight_xc_U_1[i];
        weight_xc_RUN[6] = weight_xc_U_2[i];
        weight_xc_RUN[7] = weight_xc_U_3[i];

        weight_xc_RUN += 8;
    }

    for (int i = 0; i < num_output; i++) {
        weight_hc_RUN[0] = weight_hc_R[i];
        weight_hc_RUN[1] = weight_hc_R_1[i];
        weight_hc_RUN[2] = weight_hc_R_2[i];
        weight_hc_RUN[3] = weight_hc_R_3[i];

        weight_hc_RUN[4] = weight_hc_U[i];
        weight_hc_RUN[5] = weight_hc_U_1[i];
        weight_hc_RUN[6] = weight_hc_U_2[i];
        weight_hc_RUN[7] = weight_hc_U_3[i];

        weight_hc_RUN += 8;
    }

    // 在存储N相关权重
    for (int i = 0; i < gru->input_size; i++) {
        weight_xc_RUN[0] = weight_xc_N[i];
        weight_xc_RUN[1] = weight_xc_N_1[i];
        weight_xc_RUN[2] = weight_xc_N_2[i];
        weight_xc_RUN[3] = weight_xc_N_3[i];

        weight_xc_RUN += 4;
    }

    for (int i = 0; i < num_output; i++) {
        weight_hc_RUN[0] = weight_hc_N[i];
        weight_hc_RUN[1] = weight_hc_N_1[i];
        weight_hc_RUN[2] = weight_hc_N_2[i];
        weight_hc_RUN[3] = weight_hc_N_3[i];

        weight_hc_RUN += 4;
    }
}

// 输出不能被4整除
for (; q < num_output; q++) {
    bias_c_RUBNWN[0] = bias_c_R[q];
    bias_c_RUBNWN[1] = bias_c_U[q];
    bias_c_RUBNWN[2] = bias_c_BN[q];
    bias_c_RUBNWN[3] = bias_c_WN[q];

    bias_c_RUBNWN += 4;

    const float* weight_xc_R = (float *)Mat_row(weight_xc_mat, num_output * 0 + q);
    const float* weight_xc_U = (float *)Mat_row(weight_xc_mat, num_output * 1 + q);
    const float* weight_xc_N = (float *)Mat_row(weight_xc_mat, num_output * 2 + q);

    const float* weight_hc_R = (float *)Mat_row(weight_hc_mat, num_output * 0 + q);
    const float* weight_hc_U = (float *)Mat_row(weight_hc_mat, num_output * 1 + q);
    const float* weight_hc_N = (float *)Mat_row(weight_hc_mat, num_output * 2 + q);

    float* weight_xc_RUN = (float *)Mat_row(gru->weight_xc_packed_mat, q/4 + q%4);
    float* weight_hc_RUN = (float *)Mat_row(gru->weight_hc_packed_mat, q/4 + q%4);

    for (int i = 0; i < gru->input_size; i++) {
        weight_xc_RUN[0] = weight_xc_R[i];
        weight_xc_RUN[1] = weight_xc_U[i];

        weight_xc_RUN += 2;
    }

    for (int i = 0; i < num_output; i++) {
        weight_hc_RUN[0] = weight_hc_R[i];
        weight_hc_RUN[1] = weight_hc_U[i];

        weight_hc_RUN += 2;
    }

    for (int i = 0; i < gru->input_size; i++) {
        weight_xc_RUN[0] = weight_xc_N[i];

        weight_xc_RUN += 1;
    }

    for (int i = 0; i < num_output; i++) {
        weight_hc_RUN[0] = weight_hc_N[i];

        weight_hc_RUN += 1;
    }
}
```

#### 2.5.2 重置门和更新门计算优化

对照2.2公式1，2理解下面的代码。

```c
for (; q + 3 < num_output; q += 4) {

    const float* x = (float *)Mat_row(input, t);

    // gate reset update
    const float* bias_c_RUBNWN = (const float*)gru->bias_c_packed_mat->data + q*4;

    const float* weight_xc_RUN = (float *)Mat_row(gru->weight_xc_packed_mat, q/4);
    const float* weight_hc_RUN = (float *)Mat_row(gru->weight_hc_packed_mat, q/4);

    float32x4_t _R = vld1q_f32(bias_c_RUBNWN);      // r_t bias  重置门
    float32x4_t _U = vld1q_f32(bias_c_RUBNWN + 4);  // z_t bias  更新门

    float32x4_t _sum1 = vdupq_n_f32(0.f);
    float32x4_t _sum2 = vdupq_n_f32(0.f);

    float32x4_t _sum3 = vdupq_n_f32(0.f);
    float32x4_t _sum4 = vdupq_n_f32(0.f);

    float32x4_t _sum5 = vdupq_n_f32(0.f);
    float32x4_t _sum6 = vdupq_n_f32(0.f);

    int i = 0;
    // 更新门和重置门和输入相关的计算
    // 输入能被4整除 4个输出一次性进行16次乘加操作
    for (; i + 3 < gru->input_size; i += 4) {
        float32x4_t _xi = vld1q_f32(x + i); // 输入取连续4个数据，输入连续取4个，效率更高

        // 原本每个输出需要进行256乘加操作，这里一次计算4个输出，进行4*256乘加操作
        float32x4_t _weight_xc_R = vld1q_f32(weight_xc_RUN);        // 重置门 取输入0对应的4个权重
        float32x4_t _weight_xc_U = vld1q_f32(weight_xc_RUN + 4);    // 更新门 取输入0对应的4个权重

        float32x4_t _weight_xc_R_1 = vld1q_f32(weight_xc_RUN + 8);  // 重置门 取输入1对应的4个权重
        float32x4_t _weight_xc_U_1 = vld1q_f32(weight_xc_RUN + 12); // 更新门 取输入1对应的4个权重
        
        float32x4_t _weight_xc_R_2 = vld1q_f32(weight_xc_RUN + 16); // 重置门 取输入2对应的4个权重
        float32x4_t _weight_xc_U_2 = vld1q_f32(weight_xc_RUN + 20); // 更新门 取输入2对应的4个权重
        
        float32x4_t _weight_xc_R_3 = vld1q_f32(weight_xc_RUN + 24); // 重置门 取输入3对应的4个权重
        float32x4_t _weight_xc_U_3 = vld1q_f32(weight_xc_RUN + 28); // 更新门 取输入3对应的4个权重
        
        #if __aarch64__
        _R = vfmaq_laneq_f32(_R, _weight_xc_R, _xi, 0);  
        _U = vfmaq_laneq_f32(_U, _weight_xc_U, _xi, 0);

        _sum1 = vfmaq_laneq_f32(_sum1, _weight_xc_R_1, _xi, 1);
        _sum2 = vfmaq_laneq_f32(_sum2, _weight_xc_U_1, _xi, 1);

        _sum3 = vfmaq_laneq_f32(_sum3, _weight_xc_R_2, _xi, 2);
        _sum4 = vfmaq_laneq_f32(_sum4, _weight_xc_U_2, _xi, 2);

        _sum5 = vfmaq_laneq_f32(_sum5, _weight_xc_R_3, _xi, 3);
        _sum6 = vfmaq_laneq_f32(_sum6, _weight_xc_U_3, _xi, 3);
        #else
        _R = vmlaq_lane_f32(_R, _weight_xc_R, vget_low_f32(_xi), 0);            // 输入0和权重的4个数进行乘加
        _U = vmlaq_lane_f32(_U, _weight_xc_U, vget_low_f32(_xi), 0);            // 输入0和权重的4个数进行乘加

        _sum1 = vmlaq_lane_f32(_sum1, _weight_xc_R_1, vget_low_f32(_xi), 1);    // 输入1和权重的4个数进行乘加
        _sum2 = vmlaq_lane_f32(_sum2, _weight_xc_U_1, vget_low_f32(_xi), 1);    // 输入1和权重的4个数进行乘加

        _sum3 = vmlaq_lane_f32(_sum3, _weight_xc_R_2, vget_high_f32(_xi), 0);   // 输入2和权重的4个数进行乘加
        _sum4 = vmlaq_lane_f32(_sum4, _weight_xc_U_2, vget_high_f32(_xi), 0);   // 输入2和权重的4个数进行乘加

        _sum5 = vmlaq_lane_f32(_sum5, _weight_xc_R_3, vget_high_f32(_xi), 1);   // 输入3和权重的4个数进行乘加
        _sum6 = vmlaq_lane_f32(_sum6, _weight_xc_U_3, vget_high_f32(_xi), 1);   // 输入3和权重的4个数进行乘加
        #endif

        weight_xc_RUN += 32;
    }
    // 输入不能被4整除
    for (; i < gru->input_size; i++) {
        float xi = x[i];

        float32x4_t _xi = vdupq_n_f32(xi); // 取一个输入
        float32x4_t _weight_xc_R = vld1q_f32(weight_xc_RUN);
        float32x4_t _weight_xc_U = vld1q_f32(weight_xc_RUN + 4);
        _R = vmlaq_f32(_R, _weight_xc_R, _xi);
        _U = vmlaq_f32(_U, _weight_xc_U, _xi);

        weight_xc_RUN += 8;
    }

    // 更新门和重置门和上一次输出相关的计算
    i = 0;
    // 输入能被4整除
    for (; i + 3 < num_output; i += 4) {
        float32x4_t _h_cont = vld1q_f32((const float*)hidden_state + i);
        float32x4_t _weight_hc_R = vld1q_f32(weight_hc_RUN);
        float32x4_t _weight_hc_U = vld1q_f32(weight_hc_RUN + 4);

        float32x4_t _weight_hc_R_1 = vld1q_f32(weight_hc_RUN + 8);
        float32x4_t _weight_hc_U_1 = vld1q_f32(weight_hc_RUN + 12);

        float32x4_t _weight_hc_R_2 = vld1q_f32(weight_hc_RUN + 16);
        float32x4_t _weight_hc_U_2 = vld1q_f32(weight_hc_RUN + 20);

        float32x4_t _weight_hc_R_3 = vld1q_f32(weight_hc_RUN + 24);
        float32x4_t _weight_hc_U_3 = vld1q_f32(weight_hc_RUN + 28);
        #if __aarch64__
        _R = vfmaq_laneq_f32(_R, _weight_hc_R, _h_cont, 0);
        _U = vfmaq_laneq_f32(_U, _weight_hc_U, _h_cont, 0);

        _sum1 = vfmaq_laneq_f32(_sum1, _weight_hc_R_1, _h_cont, 1);
        _sum2 = vfmaq_laneq_f32(_sum2, _weight_hc_U_1, _h_cont, 1);

        _sum3 = vfmaq_laneq_f32(_sum3, _weight_hc_R_2, _h_cont, 2);
        _sum4 = vfmaq_laneq_f32(_sum4, _weight_hc_U_2, _h_cont, 2);

        _sum5 = vfmaq_laneq_f32(_sum5, _weight_hc_R_3, _h_cont, 3);
        _sum6 = vfmaq_laneq_f32(_sum6, _weight_hc_U_3, _h_cont, 3);
        #else
        _R = vmlaq_lane_f32(_R, _weight_hc_R, vget_low_f32(_h_cont), 0);
        _U = vmlaq_lane_f32(_U, _weight_hc_U, vget_low_f32(_h_cont), 0);

        _sum1 = vmlaq_lane_f32(_sum1, _weight_hc_R_1, vget_low_f32(_h_cont), 1);
        _sum2 = vmlaq_lane_f32(_sum2, _weight_hc_U_1, vget_low_f32(_h_cont), 1);

        _sum3 = vmlaq_lane_f32(_sum3, _weight_hc_R_2, vget_high_f32(_h_cont), 0);
        _sum4 = vmlaq_lane_f32(_sum4, _weight_hc_U_2, vget_high_f32(_h_cont), 0);
        
        _sum5 = vmlaq_lane_f32(_sum5, _weight_hc_R_3, vget_high_f32(_h_cont), 1);
        _sum6 = vmlaq_lane_f32(_sum6, _weight_hc_U_3, vget_high_f32(_h_cont), 1);
        #endif

        weight_hc_RUN += 32;
    }

    // 输入不能被4整除
    for (; i < num_output; i++) {
        float h_cont = hidden_state[i];

        float32x4_t _h_cont = vdupq_n_f32(h_cont);
        float32x4_t _weight_hc_R = vld1q_f32(weight_hc_RUN);
        float32x4_t _weight_hc_U = vld1q_f32(weight_hc_RUN + 4);
        _R = vmlaq_f32(_R, _weight_hc_R, _h_cont);
        _U = vmlaq_f32(_U, _weight_hc_U, _h_cont);

        weight_hc_RUN += 8;
    }

    // 前面所有相加，计算4个输出
    _R = vaddq_f32(_R, _sum1);
    _U = vaddq_f32(_U, _sum2);
    _sum3 = vaddq_f32(_sum3, _sum5);
    _sum4 = vaddq_f32(_sum4, _sum6);
    _R = vaddq_f32(_R, _sum3);
    _U = vaddq_f32(_U, _sum4);

    // sigmoid(R)
    // sigmoid(U)
    _R = sigmoid_ps(_R);
    _U = sigmoid_ps(_U);

    ...
}
```

#### 2.5.3 当前记忆内容计算优化

对照2.2公式3理解下面的代码。

```c
for (; q + 3 < num_output; q += 4) {
    ...
    // gate new
    float32x4_t _N = vld1q_f32(bias_c_RUBNWN + 8);
    _sum1 = vdupq_n_f32(0.f);
    _sum2 = vdupq_n_f32(0.f);
    _sum3 = vdupq_n_f32(0.f);

    i = 0;
    for (; i + 3 < num_output; i += 4) {
        float32x4_t _h_cont = vld1q_f32((const float*)hidden_state + i);
        float32x4_t _weight_hc_N = vld1q_f32(weight_hc_RUN);
        float32x4_t _weight_hc_N_1 = vld1q_f32(weight_hc_RUN + 4);
        float32x4_t _weight_hc_N_2 = vld1q_f32(weight_hc_RUN + 8);
        float32x4_t _weight_hc_N_3 = vld1q_f32(weight_hc_RUN + 12);
        #if __aarch64__
        _N = vfmaq_laneq_f32(_N, _weight_hc_N, _h_cont, 0);
        _sum1 = vfmaq_laneq_f32(_sum1, _weight_hc_N_1, _h_cont, 1);
        _sum2 = vfmaq_laneq_f32(_sum2, _weight_hc_N_2, _h_cont, 2);
        _sum3 = vfmaq_laneq_f32(_sum3, _weight_hc_N_3, _h_cont, 3);
        #else
        _N = vmlaq_lane_f32(_N, _weight_hc_N, vget_low_f32(_h_cont), 0);
        _sum1 = vmlaq_lane_f32(_sum1, _weight_hc_N_1, vget_low_f32(_h_cont), 1);
        _sum2 = vmlaq_lane_f32(_sum2, _weight_hc_N_2, vget_high_f32(_h_cont), 0);
        _sum3 = vmlaq_lane_f32(_sum3, _weight_hc_N_3, vget_high_f32(_h_cont), 1);
        #endif

        weight_hc_RUN += 16;
    }

    for (; i < num_output; i++) {
        float h_cont = hidden_state[i];

        float32x4_t _h_cont = vdupq_n_f32(h_cont);
        float32x4_t _weight_hc_N = vld1q_f32(weight_hc_RUN);
        _N = vmlaq_f32(_N, _weight_hc_N, _h_cont);

        weight_hc_RUN += 4;
    }

    _N = vaddq_f32(_N, _sum1);
    _sum2 = vaddq_f32(_sum2, _sum3);
    _N = vaddq_f32(_N, _sum2);

    _N = vmlaq_f32(vld1q_f32(bias_c_RUBNWN + 12), _R, _N);
    _sum1 = vdupq_n_f32(0.f);
    _sum2 = vdupq_n_f32(0.f);
    _sum3 = vdupq_n_f32(0.f);

    i = 0;
    for (; i + 3 < gru->input_size; i += 4) {
        float32x4_t _xi = vld1q_f32(x + i);
        float32x4_t _weight_xc_N = vld1q_f32(weight_xc_RUN);
        float32x4_t _weight_xc_N_1 = vld1q_f32(weight_xc_RUN + 4);
        float32x4_t _weight_xc_N_2 = vld1q_f32(weight_xc_RUN + 8);
        float32x4_t _weight_xc_N_3 = vld1q_f32(weight_xc_RUN + 12);
        #if __aarch64__
        _N = vfmaq_laneq_f32(_N, _weight_xc_N, _xi, 0);
        _sum1 = vfmaq_laneq_f32(_sum1, _weight_xc_N_1, _xi, 1);
        _sum2 = vfmaq_laneq_f32(_sum2, _weight_xc_N_2, _xi, 2);
        _sum3 = vfmaq_laneq_f32(_sum3, _weight_xc_N_3, _xi, 3);
        #else
        _N = vmlaq_lane_f32(_N, _weight_xc_N, vget_low_f32(_xi), 0);
        _sum1 = vmlaq_lane_f32(_sum1, _weight_xc_N_1, vget_low_f32(_xi), 1);
        _sum2 = vmlaq_lane_f32(_sum2, _weight_xc_N_2, vget_high_f32(_xi), 0);
        _sum3 = vmlaq_lane_f32(_sum3, _weight_xc_N_3, vget_high_f32(_xi), 1);
        #endif

        weight_xc_RUN += 16;
    }

    for (; i < gru->input_size; i++) {
        float xi = x[i];

        float32x4_t _xi = vdupq_n_f32(xi);
        float32x4_t _weight_xc_N = vld1q_f32(weight_xc_RUN);
        _N = vmlaq_f32(_N, _weight_xc_N, _xi);

        weight_xc_RUN += 4;
    }

    _N = vaddq_f32(_N, _sum1);
    _sum2 = vaddq_f32(_sum2, _sum3);
    _N = vaddq_f32(_N, _sum2);

    // tanh(N)
    _N = tanh_ps(_N);

    float* gates_data = (float *)Mat_row(gru->gates_mat,q/4);

    vst1q_f32(gates_data, _U);
    vst1q_f32(gates_data + 4, _N);
}
```

#### 2.5.4 最终输出计算优化

对照2.2公式4理解下面的代码。

```c
// h_t := (1 - update) .* new + update .* h_{t-1}
float* output_data = (float *)Mat_row(input,t);

float* hidden_ptr = hidden_state;

q = 0;
for (; q + 3 < num_output; q += 4) {
    const float* gates_data = (float *)Mat_row(gru->gates_mat,q/4);

    float32x4_t _U = vld1q_f32(gates_data);
    float32x4_t _N = vld1q_f32(gates_data + 4);

    float32x4_t _H = vaddq_f32(vmulq_f32(vsubq_f32(vdupq_n_f32(1.f), _U), _N), vmulq_f32(_U, vld1q_f32(hidden_ptr)));

    vst1q_f32(hidden_ptr, _H);
    vst1q_f32(output_data, _H);

    hidden_ptr += 4;
    output_data += 4;
}

for (; q < num_output; q++) {
    const float* gates_data = (float *)Mat_row(gru->gates_mat, q/4 + q%4);

    float U = gates_data[0];
    float N = gates_data[1];

    float H = (1 - U) * N + U * *hidden_ptr;

    *hidden_ptr++ = H;
    *output_data++ = H;
}
```

#### 2.5.5 运行结果

GRU内部存在2个`sigmoid`和1个`tanh`计算，运算比LSTM好一些，但是也不能很好的利用多核优势。

```c
LSTM input 256 hidden 257
normal 1239us
arm_1 540us
无多核优化

GRU input 256 hidden 257
normal 940us
arm_1 420us
无多核优化
```