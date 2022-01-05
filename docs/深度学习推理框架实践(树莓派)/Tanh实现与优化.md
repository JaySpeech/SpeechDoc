# Tanh实现与优化

## 1.Tanh函数

Tanh属于`Math.h`中自带的函数。

![](img/Tanh实现与优化/tanh.jpeg ':size=30%')


## 2.Tanh Neon优化

因为时间精力，暂时未对fast tanh算法本身去做研究。

### 2.1 Fast Tanh

[fast tanh](https://codebrowser.bddppq.com/pytorch/pytorch/third_party/eigen/Eigen/src/Core/MathFunctionsImpl.h.html)

```cpp
template<typename T>
T generic_fast_tanh_float(const T& a_x)
{
  // Clamp the inputs to the range [-9, 9] since anything outside
  // this range is +/-1.0f in single-precision.
  const T plus_9 = pset1<T>(9.f);
  const T minus_9 = pset1<T>(-9.f);
  const T x = pmax(pmin(a_x, plus_9), minus_9);
  // The monomial coefficients of the numerator polynomial (odd).
  const T alpha_1 = pset1<T>(4.89352455891786e-03f);
  const T alpha_3 = pset1<T>(6.37261928875436e-04f);
  const T alpha_5 = pset1<T>(1.48572235717979e-05f);
  const T alpha_7 = pset1<T>(5.12229709037114e-08f);
  const T alpha_9 = pset1<T>(-8.60467152213735e-11f);
  const T alpha_11 = pset1<T>(2.00018790482477e-13f);
  const T alpha_13 = pset1<T>(-2.76076847742355e-16f);
  // The monomial coefficients of the denominator polynomial (even).
  const T beta_0 = pset1<T>(4.89352518554385e-03f);
  const T beta_2 = pset1<T>(2.26843463243900e-03f);
  const T beta_4 = pset1<T>(1.18534705686654e-04f);
  const T beta_6 = pset1<T>(1.19825839466702e-06f);
  // Since the polynomials are odd/even, we need x^2.
  const T x2 = pmul(x, x);
  // Evaluate the numerator polynomial p.
  T p = pmadd(x2, alpha_13, alpha_11);
  p = pmadd(x2, p, alpha_9);
  p = pmadd(x2, p, alpha_7);
  p = pmadd(x2, p, alpha_5);
  p = pmadd(x2, p, alpha_3);
  p = pmadd(x2, p, alpha_1);
  p = pmul(x, p);
  // Evaluate the denominator polynomial p.
  T q = pmadd(x2, beta_6, beta_4);
  q = pmadd(x2, q, beta_2);
  q = pmadd(x2, q, beta_0);
  // Divide the numerator by the denominator.
  return pdiv(p, q);
}

```

### 2.2 Fast Tanh NEON优化

`div_ps`实现：

```
static inline float32x4_t div_ps(float32x4_t a, float32x4_t b) {
    float32x4_t reciprocal = vrecpeq_f32(b); // finds an approximate reciprocal of each element in a vector
    reciprocal = vmulq_f32(vrecpsq_f32(b, reciprocal), reciprocal); // performs a Newton-Raphson step for finding the reciprocal.
    return vmulq_f32(a, reciprocal);
}
```

`tanh_ps`实现：

```c
#define c_tanh_tiny 1e-4f
#define c_tanh_hi   9.0f
// The monomial coefficients of the numerator polynomial (odd).
#define c_tanh_alpha_1  4.89352455891786e-3f
#define c_tanh_alpha_3  6.37261928875436e-4f
#define c_tanh_alpha_5  1.48572235717979e-5f
#define c_tanh_alpha_7  5.12229709037114e-8f
#define c_tanh_alpha_9  -8.60467152213735e-11f
#define c_tanh_alpha_11 2.00018790482477e-13f
#define c_tanh_alpha_13 -2.76076847742355e-16f
// The monomial coefficients of the denominator polynomial (even).
#define c_tanh_beta_0 4.89352518554385e-3f
#define c_tanh_beta_2 2.26843463243900e-3f
#define c_tanh_beta_4 1.18534705686654e-4f
#define c_tanh_beta_6 1.19825839466702e-6f

/* Single precision hyperbolic tangent computed for 4 simultaneous float */
static inline float32x4_t tanh_ps(float32x4_t x) {
    float32x4_t x2 = vabsq_f32(x);

    // vcgeq_f32  Vector compare Greater than or equal to
    uint32x4_t tiny_mask = vcgeq_f32(x2, vdupq_n_f32(c_tanh_tiny));

    // clamp the inputs to the range [-9, 9] since anything outside
    // this range is -/+1.0f in single-precision.
    // vbslq_u32(a,b,c) 当a=1，选择b； 当a=0，选择c
    // 当x2 < c_tanh_hi，x2不变； 当x2 > c_tanh_hi,x2=c_tanh_hi，x2不变
    x2 = vreinterpretq_f32_u32(vbslq_u32(vcgeq_f32(vdupq_n_f32(c_tanh_hi), x2), vreinterpretq_u32_f32(x2), vreinterpretq_u32_f32(vdupq_n_f32(c_tanh_hi))));

    // since the polynomials are odd/even, we need x**2.
    float32x4_t z = vmulq_f32(x2, x2);

    // evaluate the numerator polynomial y.
    float32x4_t y = vdupq_n_f32(c_tanh_alpha_13);
    y = vmlaq_f32(vdupq_n_f32(c_tanh_alpha_11), y, z);
    y = vmlaq_f32(vdupq_n_f32(c_tanh_alpha_9), y, z);
    y = vmlaq_f32(vdupq_n_f32(c_tanh_alpha_7), y, z);
    y = vmlaq_f32(vdupq_n_f32(c_tanh_alpha_5), y, z);
    y = vmlaq_f32(vdupq_n_f32(c_tanh_alpha_3), y, z);
    y = vmlaq_f32(vdupq_n_f32(c_tanh_alpha_1), y, z);
    y = vmulq_f32(y, x2);

    // evaluate the denominator polynomial w.
    float32x4_t w = vdupq_n_f32(c_tanh_beta_6);
    w = vmlaq_f32(vdupq_n_f32(c_tanh_beta_4), w, z);
    w = vmlaq_f32(vdupq_n_f32(c_tanh_beta_2), w, z);
    w = vmlaq_f32(vdupq_n_f32(c_tanh_beta_0), w, z);

    // divide the numerator by the denominator.
    #if __aarch64__
    y = vdivq_f32(y, w);
    #else
    y = div_ps(y, w);
    #endif

    // reinstate the sign.
    y = vreinterpretq_f32_u32(vbslq_u32(vdupq_n_u32(1u << 31), vreinterpretq_u32_f32(x), vreinterpretq_u32_f32(y)));

    // when the argument is very small in magnitude it's more accurate to just return it.
    y = vreinterpretq_f32_u32(vbslq_u32(tiny_mask, vreinterpretq_u32_f32(y), vreinterpretq_u32_f32(x)));

    return y;
}
```

### 2.3 Tanh Neon优化

Packing的处理方式，`size`能被4整除：

```c
int size = input->w * input->h;

if(input->elempack == 4) {
    #pragma omp parallel for num_threads(NUM_THREADS)
    for (int q = 0; q < input->c; q++) {
        float* ptr = (float *)Mat_channel(input,q);

        for (int i = 0; i < size; i++) {
            float32x4_t _p = vld1q_f32(ptr);
            _p = tanh_ps(_p);
            vst1q_f32(ptr, _p);
            ptr += 4;
        }
    }
    return 0;
}
```

实时的处理方式，先处理完能被4整除的选项：

```c
#pragma omp parallel for num_threads(NUM_THREADS)
for (int q = 0; q < input->c; q++) {
    float* ptr = (float *)Mat_channel(input,q);

    int nn = size >> 2;
    int remain = size - (nn << 2);

    for (; nn > 0; nn--) {
        float32x4_t _p = vld1q_f32(ptr);
        _p = tanh_ps(_p);
        vst1q_f32(ptr, _p);
        ptr += 4;
    }

    for (; remain > 0; remain--) {
        *ptr = tanh(*ptr);
        ptr++;
    }
}
```

### 2.4 运行结果

```c
(1000,257) 
normal 20654us  
arm_1 2291us 
多核不怎么提升
```
