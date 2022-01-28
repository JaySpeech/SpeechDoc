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

```powershell
./onnx2ncnn ncnn_test.onnx ncnn_test.param ncnn_test.bin
```

param文件内容如下，用另外一个形式表述了网络架构。

```powershell
7767517
3 3
Input            x                        0 1 x
InnerProduct     Gemm_0                   1 1 x 3 0=257 1=1 2=65792
Sigmoid          Sigmoid_1                1 1 3 y
```

在NCNN源码中加入模型测试工程：

```powershell
# 顶层CMakeLists
add_subdirectory(my_examples)
```

my_examples目录下添加自己写的测试代码和CMakeLists。

```powershell
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

### 2.2 Linear权重导入

`Linear`结构体初始化：

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

权重导入方法如下，二进制传递参数比较直接，后面可以考虑采用结构化数据进行参数传递。

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

### 2.3 Linear实现

Linear为矩阵乘法运算，实现起来比较简单。

这里假设输入为`(8,32)`，输出为`(8,16)`，权重大小为`(16,32)`。

```c
int Linear_process(Linear *linear, Mat *input, Mat* output){
    if(linear == NULL){
        return -1;
    }

    if(input->dims == 2 && output->dims ==2 && \
        input->w == linear->in_features && output->w == linear->out_features) {

        #pragma omp parallel for num_threads(NUM_THREADS)
        for (int j = 0; j < input->h; j++) {

            // 每次输入和输出各取一行数据地址
            const float* m = (float *)Mat_row(input, j);
            float* outptr = (float *)Mat_row(output, j);

            // 16次循环，每次循环进行32次乘加操作
            for (int p = 0; p < linear->out_features; p++) {
                // 取权重的一行
                const float* kptr = (const float*)linear->weight_mat->data + input->w * p;

                float sum = 0.f;

                if(linear->bias_used){
                    sum = *((float *)linear->bias_mat->data + p);
                }

                // 输入和权重进行32次乘加操作
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

### 2.4 Linear NEON优化

Linear的优化分为两种情况，一种是可以packing的，也就是`h`维度能被4整除，一种是实时性的`h`为1。

#### 2.4.1 Packing优化

```c
Mat *in = Mat_2D_create(32,8,sizeof(float),1);
float* ptr0 = (float*)in->data;
int start_num = 0;
for(int i = 0; i < in->h; i++){
    for(int j = 0; j < in->w; j++){
        *ptr0 = start_num++;
        ptr0++;
    }
}

float weight[16*32];
for(int i = 0; i < 16; i++){
    for(int j = 0; j < 32; j++){
        weight[i*32+j] = j;
        //printf("%g ",weight[i*32+j]);
    }
    //printf("\n");
}

Linear_ARM *linear = Linear_arm_create(32,16,false,true);
Linear_arm_init_variables(linear,weight,NULL);
Linear_arm_process(linear,in);
Mat*out = Linear_arm_get_output(linear);
```

首先，输入数据会进行pack。

pack前`(8,32)`：

```c
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 
32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 
64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 
96 97 98 99 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 
128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 
160 161 162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 
192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 216 217 218 219 220 221 222 223 
224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 
```

pack后`(2,32)`，每个数据块拥有4个数据，第一个数据为`0,32,64,96`，内存排布发生了变化。

```c
|0 | 1  2  3  4    5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20  21  22  23  24  25  26  27  28  29  30  31 
|32| 33 34 35 36   37  38  39  40  41  42  43  44  45  46  47  48  49  50  51  52  53  54  55  56  57  58  59  60  61  62  63 
|64| 65 66 67 68   69  70  71  72  73  74  75  76  77  78  79  80  81  82  83  84  85  86  87  88  89  90  91  92  93  94  95 
|96| 97 98 99 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 

----

128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 
160 161 162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 
192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 216 217 218 219 220 221 222 223 
224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 
```

权重内存分布：

```c
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 
```

为了加快运算，使用neon进行优化进行4float乘加操作。

```c
// in4_mat->h 2 数据已经分成两大块了
for (int j = 0; j < linear->in4_mat->h; j++) {
    // 取输出0,1数据块首地址
    float* outptr = (float *)Mat_row(linear->out4_mat,j);
    // out_features为16 该循环需要计算4行的16个结果，未优化前是一次求1行16个结果
    for (int p = 0; p < linear->out_features; p++) {
        // 0,32,64,...,480 每次32个float型，也就是权重的一行 weight_mat总长512
        const float* kptr = (const float*)linear->weight_mat->data + linear->in_features * p;
        // 取输入0,1数据块首地址
        const float* m = (float *)Mat_row(linear->in4_mat,j);
        float32x4_t _sum = vdupq_n_f32(0.f); // 将value复制4份存到返回的寄存器中

        if(linear->bias_used == true){
            _sum = vdupq_n_f32(bias_data[p]);
        }

        // in_features 32 一次循环完成4个输入数据和1个权重乘加计算，32个循环完成4个结果的输出
        for (int i = 0; i < linear->in_features; i++){
            float32x4_t _val = vld1q_f32(m);  // 从数组中依次load4个元素存到寄存器中
            float32x4_t _k = vdupq_n_f32(kptr[0]);
            _sum = vmlaq_f32(_sum, _val, _k); // 4float分别乘加

            m += 4;
            kptr += 1;
        }

        vst1q_f32(outptr, _sum); // _sum数据写到outptr中，这里为4
        outptr += 4;
    }
}
```

```c
Step1:
取输入前4行的第0个数据 0 32 64 96      取权重的第0行的第0个数据，分别做乘法
取输入前4行的第1个数据 1 33 65 97      取权重的第0行的第1个数据，分别做乘法
...
取输入前4行的第31个数据 31 63 95 127   取权重的第0行的第31个数据，分别做乘法

将上面的每一步的结果进行加法，可以得到前4行输入和权重第0行的乘加结果(4)。

Step2:
取输入前4行的第0个数据 0 32 64 96      取权重的第1行的第0个数据，分别做乘法
取输入前4行的第1个数据 1 33 65 97      取权重的第1行的第1个数据，分别做乘法
...
取输入前4行的第31个数据 31 63 95 127   取权重的第1行的第31个数据，分别做乘法

...

上述操作进行16次后，就可以得到结果(4,16)。
```

#### 2.4.2 实时优化

一般在语音场景中，pack的优化并不实用，一般语音实时场景中，`h`为1。

下面的例子输入为`(1,32)`，输出为`(1,17)`，权重为`(17,32)`。

```c
Mat *in = Mat_2D_create(32,1,sizeof(float),1);
float* ptr0 = (float*)in->data;
int start_num = 0;
for(int i = 0; i < in->h; i++){
    for(int j = 0; j < in->w; j++){
        *ptr0 = start_num++;
        ptr0++;
    }
}
float weight[17*32];
for(int i = 0; i < 17; i++){
    for(int j = 0; j < 32; j++){
        weight[i*32+j] = j;
        //printf("%g ",weight[i*32+j]);
    }
    //printf("\n");
}

Linear_ARM *linear = Linear_arm_create(32,17,false,true);
Linear_arm_init_variables(linear,weight,NULL);
Linear_arm_process(linear,in);
Mat*out = Linear_arm_get_output(linear);
```

输入矩阵内存：

```c
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 
```

权重矩阵为17行如下数据：

```c
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 
```

先来看一下权重中能被4整除的16行如何处理。

```c
const float* weight_data_ptr = (float *)linear->weight_mat->data;
float* top_blob = (float *)linear->out_mat->data;

// out_features 17  nn_num_output 4  remain_num_output_start 16
int nn_num_output = linear->out_features >> 2;
int remain_num_output_start = nn_num_output << 2;

// 每次处理4行权重
for(int pp = 0; pp < nn_num_output; pp++){

    int p = pp * 4;

    float sum0 = 0.f;
    float sum1 = 0.f;
    float sum2 = 0.f;
    float sum3 = 0.f;

    if(linear->bias_used) {
        sum0 = bias_data[p];
        sum1 = bias_data[p + 1];
        sum2 = bias_data[p + 2];
        sum3 = bias_data[p + 3];
    }

    const float* w0 = weight_data_ptr + input->w * p;           // 取权重0,4,8,12行起始地址
    const float* w1 = weight_data_ptr + input->w * (p + 1);     // 取权重1,5,9,13行起始地址
    const float* w2 = weight_data_ptr + input->w * (p + 2);     // 取权重2,6,10,14行起始地址
    const float* w3 = weight_data_ptr + input->w * (p + 3);     // 取权重3,7,11,15行起始地址

    float32x4_t _sum0 = vdupq_n_f32(0.f);
    float32x4_t _sum1 = vdupq_n_f32(0.f);
    float32x4_t _sum2 = vdupq_n_f32(0.f);
    float32x4_t _sum3 = vdupq_n_f32(0.f);

    const float* m = (float *)input->data;

    int nn = input->w >> 2;
    int remain = input->w & 3;

    // input数据每次处理4个，循环展开，一次循环进行4行权重的计算
    for(; nn > 0; nn--) {
        // 输入和权重每一行的长度都是一样的
        float32x4_t _m = vld1q_f32(m);

        float32x4_t _w0 = vld1q_f32(w0);
        _sum0 = vmlaq_f32(_sum0, _m, _w0);

        float32x4_t _w1 = vld1q_f32(w1);
        _sum1 = vmlaq_f32(_sum1, _m, _w1);

        float32x4_t _w2 = vld1q_f32(w2);
        _sum2 = vmlaq_f32(_sum2, _m, _w2);

        float32x4_t _w3 = vld1q_f32(w3);
        _sum3 = vmlaq_f32(_sum3, _m, _w3);

        m += 4;
        w0 += 4;
        w1 += 4;
        w2 += 4;
        w3 += 4;
    }

    // 不能进行neon操作的使用普通操作
    for(; remain > 0; remain--) {
        sum0 += *m * *w0;
        sum1 += *m * *w1;
        sum2 += *m * *w2;
        sum3 += *m * *w3;

        m++;
        w0++;
        w1++;
        w2++;
        w3++;
    }

    // _sum0是floatx4类型，需要将里面的4个float型相加
    // vadd_f32 float32x2_t + float32x2_t
    float32x2_t _sum0ss = vadd_f32(vget_low_f32(_sum0), vget_high_f32(_sum0)); 
    float32x2_t _sum1ss = vadd_f32(vget_low_f32(_sum1), vget_high_f32(_sum1));
    float32x2_t _sum2ss = vadd_f32(vget_low_f32(_sum2), vget_high_f32(_sum2));
    float32x2_t _sum3ss = vadd_f32(vget_low_f32(_sum3), vget_high_f32(_sum3));

    // vpadd_f32 相邻元素相加 _sum0ss两个float相加 _sum1ss两个float相加
    float32x2_t _sum01ss = vpadd_f32(_sum0ss, _sum1ss);
    float32x2_t _sum23ss = vpadd_f32(_sum2ss, _sum3ss);

    sum0 += vget_lane_f32(_sum01ss, 0);
    sum1 += vget_lane_f32(_sum01ss, 1);
    sum2 += vget_lane_f32(_sum23ss, 0);
    sum3 += vget_lane_f32(_sum23ss, 1);

    top_blob[p] = sum0;
    top_blob[p + 1] = sum1;
    top_blob[p + 2] = sum2;
    top_blob[p + 3] = sum3;        
}
```

简单梳理下上述代码的流程。

```c
Step1:
输入取 0 1 2 3   权重取第0行 0 1 2 3  乘加
                权重取第1行 0 1 2 3  乘加
                权重取第2行 0 1 2 3  乘加
                权重取第3行 0 1 2 3  乘加

Step2:
输入取 4 5 6 7   权重取第0行 4 5 6 7  乘加
                权重取第1行 4 5 6 7  乘加
                权重取第2行 4 5 6 7  乘加
                权重取第3行 4 5 6 7  乘加

...

这样就可以一个循环完成4个输出。

16行权重可以分为4次完成。

```

还剩余一行无法进行neon操作，接下来的一行如何优化呢？

```c
for (int p = remain_num_output_start; p < linear->out_features; p++) {
    float sum = 0.f;

    if(linear->bias_used)
        sum = bias_data[p];

    const float* w = weight_data_ptr + input->w * p;

    // 每次处理8个float型
    float32x4_t _sum = vdupq_n_f32(0.f);
    float32x4_t _sum2 = vdupq_n_f32(0.f);

    const float* m = (float *)input->data;

    int nn = input->w >> 3;    //整除8，这里为4
    int remain = input->w & 7;

    if(nn > 0) {
        asm volatile(
            "0:                             \n"
            "pld        [%1, #256]          \n"   // m(输入数据)预取256bytes
            "vld1.f32   {d0-d3}, [%1 :128]! \n"   // 从m(输入数据)读取8float(:128表示字节对齐)
            "pld        [%2, #256]          \n"   // 从w(权重数据)预取256bytes
            "vld1.f32   {d4-d7}, [%2]!      \n"   // 从w(权重数据)读取8float
            "vmla.f32   %q3, q0, q2         \n"   // q0存放了m(输入数据)前4float q2存放w(权重数据)前4float _sum += q0*q2
            "subs       %0, #1              \n"   // nn-=1
            "vmla.f32   %q4, q1, q3         \n"   // q1存放了m(输入数据)后4float q3存放w(权重数据)后4float _sum2 += q1*q3
            "bne        0b                  \n"
            : "=r"(nn),   // %0
            "=r"(m),    // %1
            "=r"(w),    // %2
            "=w"(_sum), // %3
            "=w"(_sum2) // %4
            : "0"(nn),
            "1"(m),
            "2"(w),
            "3"(_sum),
            "4"(_sum2)
            : "cc", "memory", "q0", "q1", "q2", "q3");  
            // "cc"表示内联汇编代码修改了标志寄存器
            // "memory"表示汇编代码对输入和输出操作数执行内存读取或写入操作（读写参数列表之一的变量指向的内存）
    }

    for(; remain > 0; remain--) {
        sum += *m * *w;

        m++;
        w++;
    }

    // float32x4_t + float32x4_t
    _sum = vaddq_f32(_sum, _sum2);

    // float32x2_t + float32x2_t
    float32x2_t _sumss = vadd_f32(vget_low_f32(_sum), vget_high_f32(_sum));

    // 两半的结果是一样的，取一半就好了
    _sumss = vpadd_f32(_sumss, _sumss);
    sum += vget_lane_f32(_sumss, 0);

    top_blob[p] = sum;
}
```

#### 2.4.3 运行结果

```c
(1,256) * (257,256) = (1,257) rt(cpu 1.5g)
normal 138us 
arm_1 55us 
arm_2 28us 
arm_4 20us

(1000,256) * (257,256) = (1000,257) packing(cpu dynamic)
normal 161614us 
arm_1 73794us 
arm_2 47922us 
arm_4 20970us
```









