# Mat实现与内存管理

## 1.1 FastMalloc

[内存对齐](https://developer.ibm.com/articles/pa-dalign/)

[c语言结构体字节对齐详解](https://segmentfault.com/a/1190000039877247)

CPU的访问粒度不仅仅是大小限制，地址上也有限制，CPU只能访问对齐地址上的固定长度的数据。
当cpu需要取4个连续的字节时，若内存起始位置的地址可以被4整除，那么我们称其对齐访问。
FastMalloc的目的是对分配的内存进行对齐，因为考虑到NEON运算，这里的对齐字节数为16字节。

Linux下提供了`posix_memalign`和`memalign`进行对齐内存分配。

```c
//一个128字节的内存块，其起始地址保证为32的倍数。
posix_memalign(&p, 32, 128)
```

下面将介绍一种通用的内存对齐的方法。

```c
#define FAST_MALLOC_ALIGN 16
#define FAST_MALLOC_OVERREAD 64

static inline unsigned char** alignPtr(unsigned char** ptr, int n) {
    return (unsigned char**)(((size_t)ptr + n - 1) & -n);
}

unsigned char* udata = (unsigned char*)malloc(size + sizeof(void*) + FAST_MALLOC_ALIGN + FAST_MALLOC_OVERREAD);
if (!udata)
    return 0;
unsigned char** adata = alignPtr((unsigned char**)udata + 1, FAST_MALLOC_ALIGN);
adata[-1] = udata;
return adata;
```

如果`fastMalloc(100)`，malloc方式是如何进行内存分配的呢？

> 由于NCNN版本不一样，之前的版本未多分配FAST_MALLOC_OVERREAD，下面的图示先忽略FAST_MALLOC_OVERREAD这一块空间。

首先malloc分配`size + sizeof(void*) + FAST_MALLOC_ALIGN`大小的内存。

![](img/Mat实现与内存管理/fastMalloc_01.png ':size=12%')

为什么要多分配 `sizeof(void *)` 大小的内存?（图中`8B`部分），因为设计者期望保存对齐前的内存指针。

为什么要多分配 `MALLOC_ALIGN` 大小的内存？（图中`16B`部分），因为对齐单位是`16(MALLOC_ALIGN)`，而对其后，数据指针不一定指在分配的内存的起始位置，
为了保证对其后仍然有`100B`目标内存可用，要`+16`。

接下来按照16字节对齐进行偏移。

![](img/Mat实现与内存管理/fastMalloc_02.png ':size=25%')

为什么要将`udata`转换成`(unsigned char **)`？`udata`地址往下跳8个字节，预留8个字节存储未对齐的地址。

```c++
// 采用类比的方式解释
unsigned char *a = new char[5];
// a + 1 表示从a指针指向的地址走了1个字节, 为啥是1个字节
// 因为unsigned char *a前面是unsigned char, unsigned char是1个字节

unsigned char **b = (unsigned char **)a;
// 同理, b + 1 表示从b指针指向的地址走了8个字节, 为啥是8个字节
// 因为unsigned char **b前面是unsigned char*, unsigned char*是8个字节
// unsigned char*是个指针类型, 指针类型占8字节的大小

// 总结: 指针的加减 可以看做去掉一个*, 然后看前面的类型占多少字节

// 所以
(unsigned char**)udata + 1
// 表示udata指向的类型为unsigned char*(是一个指针类型占8字节), 再加一, 即udata从指向的地址
// 往前走了8个字节
```

偏移的过程是寻找大于等于`ptr`且是16的整数倍的最小整数。

```c
static inline unsigned char** alignPtr(unsigned char** ptr, int n) {
    return (unsigned char**)(((size_t)ptr + n - 1) & -n);
}
```

`& -n` 如何理解？

![](img/Mat实现与内存管理/fastMalloc_03.png ':size=40%')

`n = 16 (B)`，这边的目的是让内存指向的地址为`16`的整数倍。`16`的二进制值是 `0001 0000`，`-16`的二进制值是`16`二进制值的各位取反`(...0001 1111)`，再加`1(...1111 0000)`。如果一个地址是`64`位的，这么操作后就是`0xffff ffff ffff fff0`，与目标数字按位与后，自然就是一个能被`16`整除的数字了。不仅`16`，`2^n`的数字都可以这么操作。


接下来将malloc分配的原始地址存放在对齐后的内存前面，free需要用到。

![](img/Mat实现与内存管理/fastMalloc_04.png ':size=25%')


最终内存状态。

![](img/Mat实现与内存管理/fastMalloc_05.png ':size=25%')

free时的操作：

```c
unsigned char* udata = ((unsigned char**)ptr)[-1];
free(udata);
```

## 2.Mat

* 一维矩阵，分配的总长度要求4字节的整数倍。

```c
inline size_t total(Mat *mat) {
    return mat->cstep * mat->c;
}

Mat* Mat_1D_create(int _w, size_t _elemsize){
    Mat* mat = (Mat *)malloc(sizeof(Mat));

    mat->elemsize = _elemsize;
    mat->elempack = 1;

    mat->dims = 1;
    mat->w = _w;
    mat->h = 1;
    mat->c = 1;

    mat->cstep = mat->w;

    if (total(mat) > 0){
        size_t totalsize = alignSize(total(mat) * mat->elemsize, 4);
        mat->data = fastMalloc(totalsize);
    }

    return mat;
}
```

* 二维矩阵，w和h维度的数据连续存放，分配的总长要求4字节的整数倍。

```c
Mat* Mat_2D_create(int _w, int _h, size_t _elemsize){
    Mat* mat = (Mat *)malloc(sizeof(Mat));

    mat->elemsize = _elemsize;
    mat->elempack = 1;

    mat->dims = 2;
    mat->w = _w;
    mat->h = _h;
    mat->c = 1;

    mat->cstep = (size_t)mat->w * mat->h;

    if (total(mat) > 0){
        size_t totalsize = alignSize(total(mat) * mat->elemsize, 4);
        mat->data = fastMalloc(totalsize);
    }

    return mat;
}
```

* 三维矩阵，w和h维度的数据连续存放，w和h维度的数据总长要求16字节的整数倍。w、h、c三维数据总长为4字节的整数倍。

```c
Mat* Mat_3D_create(int _w, int _h, int _c, size_t _elemsize){
    Mat* mat = (Mat *)malloc(sizeof(Mat));

    mat->elemsize = _elemsize;
    mat->elempack = 1;

    mat->dims = 3;
    mat->w = _w;
    mat->h = _h;
    mat->c = _c;

    mat->cstep = alignSize((size_t)mat->w * mat->h * mat->elemsize, 16) / mat->elemsize;

    if (total(mat) > 0){
        size_t totalsize = alignSize(total(mat) * mat->elemsize, 4);
        mat->data = fastMalloc(totalsize);
    }

    return mat;
}
```

## 3.Pack

语音数据在频域表示的维度是`(B,T,F)`，去除Batch维度，`h=T,w=F`，在非实时场景做neon优化时，需要对数据进行Pack操作。

### 3.1 二维数组Pack

```c
int test_mat(){
	Mat *in = Mat_2D_create(64,8,sizeof(float),1);
	Mat *out = Mat_2D_create(64,8/4,sizeof(float)*4,4);
	float* ptr0 = (float*)in->data;	
	int start_num = 0;
	for(int i = 0; i < in->h; i++){
		for(int j = 0; j < in->w; j++){
			*ptr0 = start_num++;
			ptr0++;
		}
	}
	Mat_2D_float_printf(in);
	Packing_2D_process(in,out);
	Mat_2D_float_printf(out);
	return 0;
}
```

Pack前内存数据：

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

Pack后内存数据：

```c
0 32 64 96 
1 33 65 97 
2 34 66 98 
3 35 67 99 
4 36 68 100 
5 37 69 101 
6 38 70 102 
7 39 71 103 
8 40 72 104 
9 41 73 105 
10 42 74 106 
11 43 75 107 
12 44 76 108 
13 45 77 109 
14 46 78 110 
15 47 79 111 
16 48 80 112 
17 49 81 113 
18 50 82 114 
19 51 83 115 
20 52 84 116 
21 53 85 117 
22 54 86 118 
23 55 87 119 
24 56 88 120 
25 57 89 121 
26 58 90 122 
27 59 91 123 
28 60 92 124 
29 61 93 125 
30 62 94 126 
31 63 95 127

128 160 192 224 
129 161 193 225 
130 162 194 226 
131 163 195 227 
132 164 196 228 
133 165 197 229 
134 166 198 230 
135 167 199 231 
136 168 200 232 
137 169 201 233 
138 170 202 234 
139 171 203 235 
140 172 204 236 
141 173 205 237 
142 174 206 238 
143 175 207 239 
144 176 208 240 
145 177 209 241 
146 178 210 242 
147 179 211 243 
148 180 212 244 
149 181 213 245 
150 182 214 246 
151 183 215 247 
152 184 216 248 
153 185 217 249 
154 186 218 250 
155 187 219 251 
156 188 220 252 
157 189 221 253 
158 190 222 254 
159 191 223 255
```

```c
int outh = out->h;                                  // 2
size_t lane_size = out->elemsize / out->elempack;   // 4

for (int i = 0; i < outh; i++) {
    // 确定packing后两大块数据的其实地址 0,512
    unsigned char* outptr = (unsigned char*)out->data + (size_t)i * in->w * out->elemsize; 

    // in->w 32, 32次完成读取和写入
    for (int j = 0; j < in->w; j++){
        // out地址 0,16,32,...,496 存4个float
        unsigned char* out_elem_ptr = outptr + j * out->elemsize;

        // elempack 4 开始从in数据4行中各取1个float
        for (int k = 0; k < out->elempack; k++){
            // srcy 
            // i = 0  0,1,2,3 
            // i = 1  4,5,6,7
            int srcy = (i * out->elempack + k) / in->elempack;
            if (srcy >= in->h)
                break;

            // srck 0
            int srck = (i * out->elempack + k) % in->elempack;

            // in->data 偏移
            // i = 0  0 128 256 384 
            // i = 1  512 640 768 896
            const unsigned char* ptr = (const unsigned char*)in->data + (size_t)srcy * in->w * in->elemsize;
            const unsigned char* elem_ptr = ptr + j * in->elemsize;
            // 从每一行中按顺序读取并赋值到out中 0,4,8,...,124
            memcpy(out_elem_ptr + k * lane_size, elem_ptr + srck * lane_size, lane_size);
        }
    }
}
```

### 3.2 二维数组Pack Neon

从上述的pack过程可以看到每次读和写都是一次一个float型，Pack过程可以借助neon进行一次4个float型读写进行加速。

先简单了解`vst4q_f32`函数的作用：

```c
typedef struct float32x4x4_t
{
    float32x4_t val[4];
}
float32x4x4_t;

vst4q_f32
Store multiple 4-element structures from four registers. 
This instruction stores multiple 4-element structures to memory from four SIMD&FP registers, 
with interleaving. Every element of each register is stored.
```

Neon Pack的流程，这里使用了`vld1q_f32`一次取

```c
// out->h 这里为2，pack后的数据由8个部分->2个部分
for (int i = 0; i < out->h; i++) {
    const float* r0 = (float *)Mat_row(in, i*4);     // 取pack前第1，5行头指针，长度32
    const float* r1 = (float *)Mat_row(in, i*4+1);   // 取pack前第2，6行头指针，长度32
    const float* r2 = (float *)Mat_row(in, i*4+2);   // 取pack前第3，7行头指针，长度32
    const float* r3 = (float *)Mat_row(in, i*4+3);   // 取pack前第4，8行头指针，长度32

    float* outptr = (float *)Mat_row(out,i);         // 取pack后第1，2行指针 pack后一行数据的长度为32*4=128

    int j = 0;

    // in->w为32，8次完成读取和写入
    for (; j + 3 < in->w; j += 4) {
        float32x4x4_t _p;
        _p.val[0] = vld1q_f32(r0);  //load r0地址的4个float到_p.val[0]
        _p.val[1] = vld1q_f32(r1);  //load r1地址的4个float到_p.val[1]
        _p.val[2] = vld1q_f32(r2);  //load r2地址的4个float到_p.val[2]
        _p.val[3] = vld1q_f32(r3);  //load r3地址的4个float到_p.val[3]
        vst4q_f32(outptr, _p);      // 将floatx4x4_t放入到数组中 从in中的4行数据各取4个float放入到outptr中(interleaving)

        r0 += 4;
        r1 += 4;
        r2 += 4;
        r3 += 4;
        outptr += 16;
    }

    // 这里未运行
    for (; j < in->w; j++) {
        outptr[0] = *r0++;  // *r0;  r0 += 1
        outptr[1] = *r1++;
        outptr[2] = *r2++;
        outptr[3] = *r3++;

        outptr += 4;
    }
}
```