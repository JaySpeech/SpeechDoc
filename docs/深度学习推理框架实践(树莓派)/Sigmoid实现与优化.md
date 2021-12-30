# Sigmoid实现与优化

## 1.Sigmoid实现

![](img/Sigmoid实现与优化/sigmoid.jpeg ':size=40%')

不做任何优化实现Sigmoid是比较简单的。

```c
for (int i = 0; i < size; i++) {
    ptr[i] = (float)(1.f / (1.f + exp(-ptr[i])));
}
```

## 2.Sigmoid Neon优化

因为时间精力，暂时未对fast exp算法本身去做研究。

### 2.1 Fast exp

[fast exp](https://root.cern/doc/v608/exp_8h_source.html)

```c
const double LOG2E = 1.4426950408889634073599; // 1/log(2)

const float MAXLOGF = 88.72283905206835f;
const float MINLOGF = -88.f;

const float C1F =   0.693359375f;
const float C2F =  -2.12194440e-4f;

const float PX1expf = 1.9875691500E-4f;
const float PX2expf = 1.3981999507E-3f;
const float PX3expf = 8.3334519073E-3f;
const float PX4expf = 4.1665795894E-2f;
const float PX5expf = 1.6666665459E-1f;
const float PX6expf = 5.0000001201E-1f;

const float LOG2EF = 1.44269504088896341f;

 /// Exponential Function single precision
 inline float fast_expf(float initial_x) {
     float x = initial_x;
 
     float z = details::fpfloor( details::LOG2EF * x +0.5f ); /* floor() truncates toward -infinity. */
 
     x -= z * details::C1F;
     x -= z * details::C2F;
     const int32_t n = int32_t ( z );
 
     const float x2 = x * x;
 
     z = x*details::PX1expf;
     z += details::PX2expf;
     z *= x;
     z += details::PX3expf;
     z *= x;
     z += details::PX4expf;
     z *= x;
     z += details::PX5expf;
     z *= x;
     z += details::PX6expf;
     z *= x2;
     z += x + 1.0f;
 
     /* multiply by power of 2 */
     z *=  details::uint322sp((n+0x7f)<<23);
 
     if (initial_x > details::MAXLOGF) z=std::numeric_limits<float>::infinity();
     if (initial_x < details::MINLOGF) z=0.f;
 
     return z;
 
   }
```

### 2.2 Fast exp NEON优化

```c
#define c_exp_hi 88.3762626647949f
#define c_exp_lo -88.3762626647949f

#define c_cephes_LOG2EF 1.44269504088896341
#define c_cephes_exp_C1 0.693359375
#define c_cephes_exp_C2 -2.12194440e-4

#define c_cephes_exp_p0 1.9875691500E-4
#define c_cephes_exp_p1 1.3981999507E-3
#define c_cephes_exp_p2 8.3334519073E-3
#define c_cephes_exp_p3 4.1665795894E-2
#define c_cephes_exp_p4 1.6666665459E-1
#define c_cephes_exp_p5 5.0000001201E-1

/* exp() computed for 4 float at once */
static inline float32x4_t exp_ps(float32x4_t x) {
    float32x4_t tmp, fx;

    float32x4_t one = vdupq_n_f32(1);
    x = vminq_f32(x, vdupq_n_f32(c_exp_hi));
    x = vmaxq_f32(x, vdupq_n_f32(c_exp_lo));

    /* express exp(x) as exp(g + n*log(2)) */
    // vmlaq_f32 0.5 + x * c_cephes_LOG2EF
    fx = vmlaq_f32(vdupq_n_f32(0.5f), x, vdupq_n_f32(c_cephes_LOG2EF)); 

    /* perform a floorf */
    // vcvtq_s32_f32 f32->s32 截断操作
    // vcvtq_f32_s32 s32->f32
    // 前面有0.5 对x * c_cephes_LOG2EF进行四舍五入操作
    tmp = vcvtq_f32_s32(vcvtq_s32_f32(fx));

    /* if greater, substract 1 */
    // vcgtq_f32 greater than 当tmp > fx时，fx = tmp-1
    uint32x4_t mask = vcgtq_f32(tmp, fx);
    // vandq_u32 logical and    vreinterpretq_u32_f32 f32->u32
    mask = vandq_u32(mask, vreinterpretq_u32_f32(one));

    // fx = tmp - mask
    fx = vsubq_f32(tmp, vreinterpretq_f32_u32(mask));

    // tmp = fx * c_cephes_exp_C1
    tmp = vmulq_f32(fx, vdupq_n_f32(c_cephes_exp_C1));
    // z = fx * c_cephes_exp_C2
    float32x4_t z = vmulq_f32(fx, vdupq_n_f32(c_cephes_exp_C2));
    // x = x - tmp
    x = vsubq_f32(x, tmp);
    // x = x - z
    x = vsubq_f32(x, z);
    // z = x * x
    z = vmulq_f32(x, x);

    float32x4_t y = vdupq_n_f32(c_cephes_exp_p0);       // y = c_cephes_exp_p0
    y = vmlaq_f32(vdupq_n_f32(c_cephes_exp_p1), y, x);  // y = c_cephes_exp_p1 + y * x
    y = vmlaq_f32(vdupq_n_f32(c_cephes_exp_p2), y, x);  // y = c_cephes_exp_p2 + y * x
    y = vmlaq_f32(vdupq_n_f32(c_cephes_exp_p3), y, x);  // y = c_cephes_exp_p3 + y * x
    y = vmlaq_f32(vdupq_n_f32(c_cephes_exp_p4), y, x);  // y = c_cephes_exp_p4 + y * x
    y = vmlaq_f32(vdupq_n_f32(c_cephes_exp_p5), y, x);  // y = c_cephes_exp_p5 + y * x

    y = vmlaq_f32(x, y, z); // y = x + y * z
    y = vaddq_f32(y, one);  // y = y + 1

    /* build 2^n */
    int32x4_t mm;
    mm = vcvtq_s32_f32(fx);                // mm = int(fx)
    mm = vaddq_s32(mm, vdupq_n_s32(0x7f)); // mm = mm + 0x7f
    mm = vshlq_n_s32(mm, 23);              // mm = mm << 23
    float32x4_t pow2n = vreinterpretq_f32_s32(mm); // pow2n = (float)mm

    y = vmulq_f32(y, pow2n); // y = y * pow2n
    return y;
}
```

### 2.3 Sigmoid Neon优化

借助exp_ps进行sigmoid_ps实现。

```c
static inline float32x4_t sigmoid_ps(float32x4_t _v) {
    float32x4_t _one = vdupq_n_f32(1.f);
    _v = vnegq_f32(_v);    // -_v = _v
    _v = exp_ps(_v);
    _v = vaddq_f32(_v, _one);
    float32x4_t _outp = vrecpeq_f32(_v);  // 倒数 此时能达到千分之一左右的精度
    return vmulq_f32(vrecpsq_f32(_v, _outp), _outp); // 执行后能达到百万分之一左右的精度
}
```

Packing的处理方式：

```c
int size = input->w * input->h;

for (int q = 0; q < input->c; q++) {
    float* ptr = (float*)Mat_channel(input,q);

    // 每次读取4float进行处理
    for (int i = 0; i < size; i++) {
        float32x4_t _p = vld1q_f32(ptr);
        _p = sigmoid_ps(_p);
        vst1q_f32(ptr, _p);

        ptr += 4;
    }
}
```

实时的处理方式：

```c
int size = input->w * input->h;

for (int q = 0; q < input->c; q++) {
    float* ptr = (float*)Mat_channel(input,q);

    int nn = size >> 2;
    int remain = size - (nn << 2);

    for(; nn > 0; nn--) {
        float32x4_t _p = vld1q_f32(ptr);
        _p = sigmoid_ps(_p);
        vst1q_f32(ptr, _p);
        ptr += 4;
    }

    for (; remain > 0; remain--) {
        *ptr = 1.f / (1.f + exp(-*ptr));
        ptr++;
    }
}
```

### 2.4 运行结果

```c
(1000,257)
normal 12067us  arm_1 3825us 多核不怎么提升
```
