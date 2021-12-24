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

一维矩阵，分配的总长度要求4字节的整数倍。

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

二维矩阵，w和h维度的数据连续存放，分配的总长要求4字节的整数倍。

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

三维矩阵，w和h维度的数据连续存放，w和h维度的数据总长要求16字节的整数倍。w、h、c三维数据总长为4字节的整数倍。

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





