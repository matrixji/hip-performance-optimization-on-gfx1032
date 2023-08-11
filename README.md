# 写在前面

此仓库用于复现[fsword73/HIP-Performance-Optmization-on-VEGA64](https://github.com/fsword73/HIP-Performance-Optmization-on-VEGA64)
在我自己机器上的运行结果，算是了解和学习AMD GPU性能相关主题的一个学习笔记，也希望能做为一个在AMD消费级显卡上进行HIP编程的参考。

本文从第二章起基本完全复现原文的实验，但是会对代码做一些优化，增加了CMake的工程，方便阅读和复现。所有的测试在ROCm 5.6.0 + RX6600上完成。

## 关于HIP和ROCm

关于这部分的知识建议参考 [fsword73/HIP-Performance-Optmization-on-VEGA64](https://github.com/fsword73/HIP-Performance-Optmization-on-VEGA64) 第一章节的介绍。由于ROCm的文档并没有CUDA那么详细，同时社区用户也不在一个数量级，因此参考资料能起多大的作用我也不好说。

## 关于硬件
我有一块二手的 Radeon RX 6600的显卡 gfx1032，最初买来尝试移植[faiss](https://github.com/facebookresearch/faiss)到 ROCm上，目前基本代码移植OK了，但是单元测试遇到了很多结果不预期的问题，目前还在排查中。希望能在未来的某个时间点能够解决这些问题，然后也可以给大家开放这部分的移植代码。

Radeon RX 6600 属于RDNA2的架构，ROCm也是从5.6 开始有了比较好的支持，但是实际上应该还是有很多的问题。这里不一一赘述了，总而言之，要想学习GPU编程或者深度学习还是用NVIDIA的卡吧，没必要和自己过不去。当然也希望ROCm社区能够加强软件的成熟度，也能为大家提供一个除了CUDA之外的选择。

## 硬件频率
原文使用的是Mi25的显卡，锁定了1.536Ghz的频率。但是RDNA的卡光是锁频这件事情，就很麻烦，幸好有大神写了脚本 [sibradzic/amdgpu-clocks](https://github.com:sibradzic/amdgpu-clocks.git) 我们拿来使用就行了，主要分两步：

1. 通过修改Grub参数让显卡支持手动设置频率，Kernel启动参数需要加上 `amdgpu.ppfeaturemask=0xfff7ffff` 然后重新生成Grub，重启即可，配置成功后，你应该能够看到 `/sys/class/drm/card0/device/pp_od_clk_voltage` 文件是存在的。
2. 配置一个 `/etc/default/amdgpu-custom-state.card0` 的配置文件，来设置SCLK，在使用 `amdgpu-clock` 来锁频。

下面是我的配置文件，可以参考一下：

```bash
# /etc/default/amdgpu-custom-state.card0
OD_SCLK:
0: 1000MHz
1: 1000MHz
# Force power limit (in micro watts):
FORCE_POWER_CAP: 87000000
FORCE_PERF_LEVEL: manual
```

为了方便计算我把频率设置成了1000Mhz。设置过程中能发现 RX6600 似乎最高支持2900MHz，算了不敢瞎设置，卡坏了，得自己掏钱买新卡。

本文的测试均在锁频1000Mhz状态下进行。如果你自己使用本代码进行复现，那么可以修改 [const.h](src/const.h) 中的变量 `kSclk`


# 基于RX6600的硬件相关优化实例
## 块与线程
### 最高线程速率

AMD RDNA硬件约定32 Threads 一个 wave，一个block可以有1-32个wave。当然原文的Mi25是GCN架构，约定是 64 * 1～16。
当然hipcc 有一个编译选项 `-mno-wavefrontsize64` 和  `-mwavefrontsize64`，但是实际上不要报太大希望，我测试感觉他只是一个语法糖，在Launch线程做了处理。硬件上的差异还是存在的。

硬件生成Threads的速率将直接影响最终程序的效率，例如GPU显存的读写速度。 为了测试显卡的Threads速率， 我们可以写一个最简单的设备空函数。

```cpp
__global__ void null_kernel(float * __restrict__ a) {

}
```

因此程序设置总的Threads 数量为 1024 * 1204 * 1024, 已获得接近秒级的GPU执行时间。
Threads速率是否与Block速率相关？这仍然是一个谜。因此测试程序暂时将每个 Block的Threads设置为最大值 1024。

为了获得准备的时间， 使用hipEventCreate函数产生两个事件 start, stop,通过hipEventRecord记录两个事件，并调用hipEventSynchronize确保stop是同步事件并被正确执行，hipEventElapsedTime(&elapsed_time_ms, start, stop)函数将获得start, stop两个event的时间长度，单位是毫秒。代码如下：

```cpp
HIP_CHECK(hipEventRecord(start));
null_kernel<<<..., ...>>>(a);
HIP_CHECK(hipEventRecord(stop));
HIP_CHECK(hipEventSynchronize(stop));
HIP_CHECK(hipEventElapsedTime(&elapsed_time_ms, start, stop));
```

HIP_CHECK 一个方便检查Hip相关API返回值的宏，他的定义在 [utils.h](src/utils.h) 中。

完成的代码参考：[test-02-01.hip](src/test-02-01.hip)，运行一下，结果如下（注意一定不要是Debug模式，性能影响很大）：
```plain
Device name: AMD Radeon RX 6600
System major: 10
System minor: 3
Elapsed time: 25.3723 ms
Totals: 1073741824threads, dim3(1024, 1, 1) 42.3195 Threads/Cycle
```

大概是42 Threads/Cycle。 并没有获得原文GCN架构中约64 Threads/Cycle的速率。

### 1D形状 Block的线程速率曲线
线程速率为 64 threads/cycle,那么是不是所有1D 形状块均可获得极限速率呢？
我们通过参数来测试不同的块，得到如下结果：
```plain
Totals: 1073741824threads, dim3(1, 1, 1) 0.995889 Threads/Cycle
Totals: 1073741824threads, dim3(2, 1, 1) 1.99163 Threads/Cycle
Totals: 1073741824threads, dim3(4, 1, 1) 3.98346 Threads/Cycle
Totals: 1073741824threads, dim3(8, 1, 1) 7.96135 Threads/Cycle
Totals: 1073741824threads, dim3(16, 1, 1) 15.9146 Threads/Cycle
Totals: 1073741824threads, dim3(32, 1, 1) 31.7863 Threads/Cycle
Totals: 1073741824threads, dim3(64, 1, 1) 42.2529 Threads/Cycle
Totals: 1073741824threads, dim3(128, 1, 1) 42.3673 Threads/Cycle
Totals: 1073741824threads, dim3(256, 1, 1) 42.4139 Threads/Cycle
Totals: 1073741824threads, dim3(512, 1, 1) 42.1957 Threads/Cycle
Totals: 1073741824threads, dim3(1024, 1, 1) 42.4215 Threads/Cycle
```

> 仔细观察，仅仅当 BlockDim = 256， 512, 1024时， 线程产生速度达到峰值。这个信息有什么含义， 或者对GPU程序优化有何指导意义？
> 举例， 在深度学习中有大量的简单操作， 例如Copy, 激活函数，如果程序使用了比256小的BlockDim, 那么程序将很难达到理论值, 例如64，那么理论极限很有可能> 是64/256。深度学习经常使用Padding Copy, 如果 H x W = 7x7, Padding= 3, 那么理论极限将会是13*13/256 = 66%。
> 以上两种情况， 如果程序能够将原来4 threads的工作合并到一个thread，每个线程处理的事务随之提高到4倍，例如读写操作，将极大地提高理论极限。

> 这个测试结果是否有值得怀疑的地方？ 这个测试结果证明只有BlockDim =256才能达到理论极限，和AMD GCN的图形像素渲染能力不匹配，颜色渲染能力达到了64 Pixels/Cycle。GCN架构的Pixel Shader都是64个 像素一个Wave，换而言之HIP 也应该能够达到64 Threads/Cycle。而测试结果只有Pixel Shader的1/4，这有两种可能： 1） ROCm使用了特别的寄存器设置使得线程产生速度降低到了1/4；2）硬件的计算线程生成速度是像素着色器的1/4速度。第二个原因的可能性比较小，GCN统一化的着色器架构设计应保证不同类型的着色器（几何，像素，计算）线程速度相同， 否则对应硬件资源将被浪费。

上述是原文中关于这一结果展开解释和分析。个人认为最重要的一点知道Block Dim在256整数倍的时候能够达到最佳的线程生成速率就足够指导一部分的程序优化工作了。当然这上面说的是GCN的架构，至少在RX6600上我们现在测试得到的结论是64的整数倍最佳。

### 2D 形状Block线程速率
如法炮制，我们组合不同的BlockDim，得到如下结果：
```plain
Totals: 1073741824threads, dim3(1, 1, 1) 0.995783 Threads/Cycle
Totals: 1073741824threads, dim3(2, 2, 1) 3.98216 Threads/Cycle
Totals: 1073741824threads, dim3(4, 4, 1) 15.9111 Threads/Cycle
Totals: 1073741824threads, dim3(8, 8, 1) 42.3601 Threads/Cycle
Totals: 1073741824threads, dim3(16, 16, 1) 42.385 Threads/Cycle
Totals: 1073741824threads, dim3(32, 32, 1) 42.3749 Threads/Cycle
```

同样只有在64的整数倍的时候才能达到最佳的线程生成速率。

### 3D 形状Block的线程生成速率
如法炮制，我们组合不同的BlockDim，得到如下结果：
```plain
Totals: 1073741824threads, dim3(1, 1, 1) 0.996201 Threads/Cycle
Totals: 1073741824threads, dim3(2, 2, 2) 7.96841 Threads/Cycle
Totals: 1073741824threads, dim3(3, 3, 3) 26.8356 Threads/Cycle
Totals: 1073741824threads, dim3(4, 4, 4) 42.3864 Threads/Cycle
Totals: 1073741824threads, dim3(5, 5, 5) 41.3835 Threads/Cycle
Totals: 1073741824threads, dim3(6, 6, 6) 40.7182 Threads/Cycle
Totals: 1073741824threads, dim3(7, 7, 7) 41.0964 Threads/Cycle
Totals: 1073741824threads, dim3(8, 8, 8) 42.3944 Threads/Cycle
Totals: 1073741824threads, dim3(9, 9, 9) 41.9737 Threads/Cycle
Totals: 1073741824threads, dim3(10, 10, 10) 41.3719 Threads/Cycle
```

所以说3D BlockDim也只有在64的整数倍的时候才能达到最佳的线程生成速率。

> 对于HIP程序开发者，对于简单的显存读写类，建议使用256倍数的BlockDim以获取最高线程生成速率。计算异常密集的任务，它的性能主要瓶颈和线程生成速率无关时，建议使用64倍数的BlockDim。

### 思考

对于CUDA来说，上面的结论该是如何？RTX的卡和Tesla系列是否有差别？我们需要去实践测试验证了。