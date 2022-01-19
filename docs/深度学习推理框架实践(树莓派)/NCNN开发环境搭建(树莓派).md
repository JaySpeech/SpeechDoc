# NCNN开发环境搭建(树莓派)

## 1.OpenCV编译

下载openCV：

```powershell
https://github.com/Itseez/opencv/archive/2.4.13.zip
```

编译依赖：

```powershell
#安装编译工具
sudo apt-get install build-essential
#安装依赖包
sudo apt-get install cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
#安装可选包
sudo apt-get install python-dev python-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev
```

编译安装：

```powershell
#打开文件夹"opencv-2.4.13"：
cd opencv-2.4.13

#新建一个文件夹用于存放临时文件：
mkdir release

#切换到该临时文件夹：
cd release

#开始编译：
cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local ..
make -j4    //开启线程 按照自己的配置
sudo make install
```

在树莓派上编译会出现一些错误，按照提示的修改方法即可。

```makefile                   
error: ‘CODEC_FLAG_GLOBAL_HEADER’ was not declared in this scope                                                                 
note: suggested alternative: ‘AV_CODEC_FLAG_GLOBAL_HEADER’

error: ‘AVFMT_RAWPICTURE’ was not declared in this scope                                                                   
note: suggested alternative: ‘AVFMT_NOFILE’ 
```

安装后会生成如下文件：

```powershell
ls /usr/local/lib/ 
libopencv_calib3d.so            libopencv_highgui.so.2.4       libopencv_ocl.so.2.4.13
libopencv_calib3d.so.2.4        libopencv_highgui.so.2.4.13    libopencv_photo.so
libopencv_calib3d.so.2.4.13     libopencv_imgproc.so           libopencv_photo.so.2.4
libopencv_contrib.so            libopencv_imgproc.so.2.4       libopencv_photo.so.2.4.13
libopencv_contrib.so.2.4        libopencv_imgproc.so.2.4.13    libopencv_stitching.so
libopencv_contrib.so.2.4.13     libopencv_legacy.so            libopencv_stitching.so.2.4
libopencv_core.so               libopencv_legacy.so.2.4        libopencv_stitching.so.2.4.13
libopencv_core.so.2.4           libopencv_legacy.so.2.4.13     libopencv_superres.so
libopencv_core.so.2.4.13        libopencv_ml.so                libopencv_superres.so.2.4
libopencv_features2d.so         libopencv_ml.so.2.4            libopencv_superres.so.2.4.13
libopencv_features2d.so.2.4     libopencv_ml.so.2.4.13         libopencv_ts.a
libopencv_features2d.so.2.4.13  libopencv_nonfree.so           libopencv_video.so
libopencv_flann.so              libopencv_nonfree.so.2.4       libopencv_video.so.2.4
libopencv_flann.so.2.4          libopencv_nonfree.so.2.4.13    libopencv_video.so.2.4.13
libopencv_flann.so.2.4.13       libopencv_objdetect.so         libopencv_videostab.so
libopencv_gpu.so                libopencv_objdetect.so.2.4     libopencv_videostab.so.2.4
libopencv_gpu.so.2.4            libopencv_objdetect.so.2.4.13  libopencv_videostab.so.2.4.13
libopencv_gpu.so.2.4.13         libopencv_ocl.so               pkgconfig
libopencv_highgui.so            libopencv_ocl.so.2.4                   

ls -l /usr/local/include/opencv
total 44
-rw-r--r-- 1 root root 3438 Feb 21  2018 cv.h
-rw-r--r-- 1 root root 2411 Feb 21  2018 cv.hpp
-rw-r--r-- 1 root root 2850 Feb 21  2018 cvaux.h
-rw-r--r-- 1 root root 2346 Feb 21  2018 cvaux.hpp
-rw-r--r-- 1 root root 2184 Feb 21  2018 cvwimage.h
-rw-r--r-- 1 root root 2465 Feb 21  2018 cxcore.h
-rw-r--r-- 1 root root 2423 Feb 21  2018 cxcore.hpp
-rw-r--r-- 1 root root 2265 Feb 21  2018 cxeigen.hpp
-rw-r--r-- 1 root root 2387 Feb 21  2018 cxmisc.h
-rw-r--r-- 1 root root 2306 Feb 21  2018 highgui.h
-rw-r--r-- 1 root root 2189 Feb 21  2018 ml.h

ls  /usr/local/include/opencv2/
calib3d      features2d      highgui      ml             ocl                 stitching      ts.hpp
calib3d.hpp  features2d.hpp  highgui.hpp  ml.hpp         opencv.hpp          stitching.hpp  video
contrib      flann           imgproc      nonfree        opencv_modules.hpp  superres       video.hpp
core         flann.hpp       imgproc.hpp  objdetect      photo               superres.hpp   videostab
core.hpp     gpu             legacy       objdetect.hpp  photo.hpp           ts             videostab.hpp

cat /usr/local/lib/pkgconfig/opencv.pc 
# Package Information for pkg-config

prefix=/usr/local
exec_prefix=${prefix}
libdir=${exec_prefix}/lib
includedir_old=${prefix}/include/opencv
includedir_new=${prefix}/include

Name: OpenCV
Description: Open Source Computer Vision Library
Version: 2.4.13.6
Libs: -L${exec_prefix}/lib -lopencv_calib3d -lopencv_contrib -lopencv_core -lopencv_features2d -lopencv_flann -lopencv_gpu -lopencv_highgui -lopencv_imgproc -lopencv_legacy -lopencv_ml -lopencv_nonfree -lopencv_objdetect -lopencv_ocl -lopencv_photo -lopencv_stitching -lopencv_superres -lopencv_ts -lopencv_video -lopencv_videostab -lrt -lpthread -lm -ldl
Cflags: -I${includedir_old} -I${includedir_new}
```

pkgconfig配置：

```powershell
#配置环境

#将opencv的库加入到路径，从而让系统可以找到
sudo gedit /etc/ld.so.conf.d/opencv.conf

#末尾加入/usr/local/lib，保存退出
sudo ldconfig    使配置生效

sudo gedit /etc/bash.bashrc 
#末尾加入
PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/usr/local/lib/pkgconfig
export PKG_CONFIG_PATH

#保存退出
sudo source /etc/bash.bashrc  #使配置生效
#（该步骤可能会报错找不到命令，原因是source为root命令su（进入root权限）

#输入密码
source /etc/bash.bashrc
Ctrl+d（推迟root）

#sudo updatedb #更新database
```

测试：

```cpp
//文件名字Dis.cpp
#include <stdio.h>
#include <opencv2/opencv.hpp>
using namespace cv;
int main( )
{
    Mat image;
    image = imread("/home/elijah/lena.jpg", 1 );//目录按照自己的目录
    if ( !image.data )
    {
        printf("No image data \n");
        return -1;
    }
    namedWindow("Display Image", WINDOW_AUTOSIZE );
    imshow("Display Image", image);
    waitKey(0);
    return 0;
}
```

编译测试：

```powershell
g++ Dis.cpp -o Dis `pkg-config --cflags --libs opencv`
```

## 2.NCNN编译

### 2.1 NCNN编译流程

下载NCNN：

```powershell
git clone https://github.com/Tencent/ncnn.git
```

依赖环境：

```powershell
* g++
* cmake
* protocol buffer (protobuf) headers files and protobuf compiler
* vulkan header files and loader library
* (optional) opencv # For building examples
```

在ubuntu、树莓派官方推荐安装方法(不考虑)：

```powershell
sudo apt install build-essential git cmake libprotobuf-dev protobuf-compiler libvulkan-dev vulkan-utils libopencv-dev
```

在这里暂时我们不使用vulkan，并且opencv使用源码的方式安装，安装命令如下：

```powershell
sudo apt install build-essential git cmake libprotobuf-dev protobuf-compiler
```

ubuntu编译命令：

```powershell
cd ncnn
mkdir build && cd build
cmake ..
make -j4
make install
```

树莓派编译命令：

```powershell
cd <ncnn-root-dir>
mkdir -p build
cd build
cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/pi3.toolchain.cmake -DPI3=ON ..
make -j4
make install
```

编译发生如下错误：

```powershell
undefined reference to symbol 'dlopen@@GLIBC_2.4'
```

修改CMakeLists增加dl依赖即可：

```powershell
if(PI3)
    target_compile_options(ncnn PRIVATE -march=native -mfpu=neon -mfloat-abi=hard)
    target_compile_definitions(ncnn PRIVATE __ARM_NEON __ANDROID__)
    target_link_libraries(ncnn PUBLIC dl)
endif()
```

### 2.2 NCNN在arm上的编译选项

pi3.toolchain.cmake内容：

```makefile
SET(CMAKE_SYSTEM_NANE Android)
SET(CMAKE_SYSTEM_PROCESSOR "armv7l")
SET(ANDROID_ARCH_NAME "arm")
SET(UNIX true)
SET(CMAKE_C_COMPILER "gcc")
SET(CMAKE_CXX_COMPILER "g++")
```

针对PI3 ARM平台，编译的时候会打开neon：

```makefile
if(PI3)
    target_compile_options(ncnn PRIVATE -march=native -mfpu=neon -mfloat-abi=hard)
    target_compile_definitions(ncnn PRIVATE __ARM_NEON __ANDROID__)
endif()
```

ncnn_add_layer.cmake根据平台自动生成layer_declaration.h：

```cpp
namespace ncnn {
class InnerProduct_final : virtual public InnerProduct, virtual public InnerProduct_arm
{
public:
    virtual int create_pipeline(const Option& opt) {
        { int ret = InnerProduct::create_pipeline(opt); if (ret) return ret; }
        { int ret = InnerProduct_arm::create_pipeline(opt); if (ret) return ret; }
        return 0;
    }
    virtual int destroy_pipeline(const Option& opt) {
        { int ret = InnerProduct_arm::destroy_pipeline(opt); if (ret) return ret; }
        { int ret = InnerProduct::destroy_pipeline(opt); if (ret) return ret; }
        return 0;
    }
};
DEFINE_LAYER_CREATOR(InnerProduct_final)
```
