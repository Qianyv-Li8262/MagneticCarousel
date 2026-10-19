# MagneticCarousel

一个起源于普通物理实验课程的磁转盘模拟项目，题干见 2026 IYPT 第16题，研究旋转磁铁阵列驱动钢球运动的过程。

模型将球心运动限制在平面内，使用中点欧拉法，同时跟踪球的三维转动、磁滞记忆和涡流响应。核心使用 C++ / CUDA 实现，通过 Python 运行轨迹、绘制动画和扫描参数，用于研究启动、绕行、自旋以及滑动与纯滚动之间的转换。

项目仍在研究中，模拟与实验之间存在尚未解释的差异。目前独立推进遇到了一些瓶颈，希望通过开源，与感兴趣的人共同检查物理模型、数值实现和实验分析，寻找进一步改进的方向。

## 目前的局限

- **尚未同时复现几个关键现象。** 以往拟合中，减小半径过冲往往难以兼顾低速等待与快速启动；接近自旋目标的候选参数，也未必能同时匹配绕行周期。目前尚无经过充分验证的参数组，能够同时解释这些现象。
- **自旋测量仍有不确定性。** 球面标记存在遮挡和运动模糊，采样混叠与相机标定也会影响结果。经过我们的实验，我认为自转尤其的难以靠视频测定……
- **部分参数是条件拟合值。** 磁性参数、电导率和滚阻并非全部来自这批钢球的独立测量，不同参数可能互相补偿。滚阻随速度变化的规律也尚未得到可靠确认。
- **物理近似还需要检验。** 当前将磁滞响应与固定标量磁导率下的线性涡流增量结合；工作状态下的增量磁导率、两者的完整耦合，以及球内场不均匀的影响，仍值得检查。

## 构建与运行

以下命令从包含本 README 和 `CMakeLists.txt` 的目录执行。

需要 Python 3.10 或更新版本、支持 C++17 的编译环境、CMake、Ninja、Boost.Multiprecision，以及可用的 OpenMP。CPU 构建要求 CMake ≥ 3.24，CUDA 构建要求 ≥ 3.27。

硬件上， CUDA 要求你有一张N卡，没有使用高架构的特性，所以我认为比较老旧的也能跑，只是性能问题会稍显严重。如果性能太差可以加粗步长……

CPU 方面，需要支持硬件 AVX2 的向量化。代码里显式使用了 AVX2 进行推进。

先在准备运行程序的 Python 环境里安装：

```powershell
python -m pip install numpy matplotlib pybind11
```

### 先运行 CPU 版本

在已准备好编译器和 Boost 的环境中配置、构建：

```powershell
cmake -S . -B build/release -G Ninja -DCMAKE_BUILD_TYPE=Release -DIYPT16_ENABLE_CUDA=OFF -DBUILD_TESTING=OFF
cmake --build build/release --parallel 6
```

如果 Boost 由 vcpkg 提供，首次配置时还需传入自己的 `-DCMAKE_TOOLCHAIN_FILE=...`；也可以通过 `CMAKE_PREFIX_PATH` 指定依赖安装位置。用 `-DPython_EXECUTABLE=...` 可以明确指定 Python，构建与运行应使用同一环境。

默认使用有限圆柱磁铁模型，构建会自动生成磁场 LUT 和涡流系数，不需要手工运行生成器。CPU 版本不需要 NVIDIA GPU。

构建后先检查配置，再打开动画：

```powershell
python animate_one_trajectory.py --dry-run
python animate_one_trajectory.py
```

默认使用 FP64；加 `--kahan` 可选择带 Kahan 补偿的 FP32 版本。更多运行选项见 `python animate_one_trajectory.py --help`。

### CUDA 与本机配置

GPU 扫描需要一张N卡、兼容的驱动与 CUDA Toolkit，以及与 Toolkit 匹配的 CuPy。Windows 下还需要 MSVC 和 Windows SDK；首次配置应在相应开发环境中执行。

在 CPU 配置命令中改用 `-DIYPT16_ENABLE_CUDA=ON`，并按 GPU 设置 `CMAKE_CUDA_ARCHITECTURES`，即可构建 CUDA 扫描程序。例如在本机有可检测的兼容 GPU 时，可以使用 `-DCMAKE_CUDA_ARCHITECTURES=native`。

本项目目前已在作者的 Windows x64 / Clang 22 / CUDA 13.3 环境完成从空构建目录到 CPU、CUDA 产物生成的验证；其他工具链组合尚未逐一验证。上面的通用配置仍需按自己的依赖安装位置调整。

作者本机使用下面的 preset 命令。`CMakeUserPresets.json` 含机器相关路径，不随仓库分发；自己的配置准备好后才可使用。名称虽然带 `cpu`，当前本机配置实际同时启用了 CUDA。

```powershell
cmake --preset local-cpu
cmake --build --preset local-cpu-release
```

## 参数与代码入口

调参前请阅读[无量纲约定、参数换算与模型说明](docs/PARAMETERS_AND_MODEL.md)。其中列出了 SI 换算、一维与二维单位的区别、初态和扫描字段的含义，以及线性化矢量 JA、涡流响应近似和 Fourier–spline LUT 的实现形式。

| 内容 | 入口 |
| --- | --- |
| 单条轨迹、初态和动画参数 | `animate_one_trajectory.py`，先看 `--help` |
| 初态扫描 | `scan_initial_2d.py`，支持 CPU / GPU |
| 球径与盘转速扫描 | `scan_sr_somega_2d_gpu.py` |
| 磁滞、摩擦与滚阻参数 | `kernels/headers/hysteresis_config.hpp` |
| 磁场高度、磁铁数量与 LUT 分辨率 | `kernels/headers/hysteresis_config_field.hpp` |
| SI 参考量与电导率 | `kernels/headers/hysteresis_physics.hpp` |
| 动力学和时间推进 | `kernels/headers/2dmodel-hysteresis.hpp` |
| 涡流系数生成方法 | [EDDY_CODEGEN.md](kernels/tools/EDDY_CODEGEN.md) |
| 数值检查 | `tests/` |

修改 C++ / CUDA 源码或编译期参数后，重新执行：

```powershell
cmake --build build/release --parallel 6
```

修改一般动力学参数不会重新生成磁场 LUT；修改磁场几何、LUT 配置或生成器则会。涡流系数生成器目前依赖整份材料配置头文件，因此部分与涡流无关的参数修改也会触发系数重生成。

球径的运行时倍率不会自动修改编译期磁场高度。更换钢球或调整装置后，请核对球心到磁铁中心的实际距离；此前的试算曾因这一换算产生偏差。

构建产物位于 `build/release/`，运行数据通常在 `runs/`，图像在 `scratch/`。`cmake --fresh` 重新配置缓存，不等于删除全部构建产物。

## 交流与贡献

欢迎讨论模型与实验之间的差异，尤其是材料磁化规律、磁滞与涡流的耦合，以及装置标定和自旋重建中的不确定性。也欢迎对动力学推导和数值实现进行独立检查，提供对照计算、问题复现或改进建议。

提交问题时，请尽量附上代码版本、运行命令、参数、构建环境及相关输出。涉及实验对比时，请同时说明球径、球心到磁铁中心的距离、磁铁盘转速及数据处理方法，以便复现和分析。

## 许可证

源码采用 [GNU AGPL-3.0-only](LICENSE)。
