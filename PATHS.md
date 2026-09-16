# 项目路径约定

所有活跃脚本的默认路径均以 Git 仓库根目录为基准，不再依赖运行命令时的当前目录。

| 目录 | 用途 | 是否进入 Git |
|---|---|---|
| `build/release/` | CMake Release 构建树，以及 Python 扩展、链接库和可执行文件 | 否 |
| `runs/` | 原始扫描、轨迹、日志和可再生成数据 | 否 |
| `results/` | 汇总表、报告文本和精选数值结果；程序不向这里生成图像 | 是 |
| `scratch/` | 所有程序生成的图像、smoke test、一次性诊断和临时文件 | 否 |
| `tmp/` | 尚未完成归档的历史数据，只用于兼容读取 | 否 |

## 源码布局

Python 入口和分析程序保留在 `src/` 范围内。主模型的 C++/CUDA 实现集中在
`kernels/`，Python 扩展绑定放在 `kernels/bindings/`，公共模拟接口放在
`kernels/headers/`。Kahan 累加和 double-double 等可复用数值实现位于
`kernels/headers/numeric_extensions/`，原生代码测试位于 `tests/native/`。

`toymodel/` 是独立的玩具模型，其中的 CUDA 文件与对应 Python 驱动共同保留，
不并入主模型内核目录。

## 可配置目录

可以设置 `IYPT16_BUILD_DIR`、`IYPT16_RUNS_DIR`、`IYPT16_RESULTS_DIR` 或
`IYPT16_SCRATCH_DIR`，把大数据放到其他磁盘而无需修改源码。其中
`IYPT16_BUILD_DIR` 应直接指向 CMake binary directory（默认是 `build/release/`）。
命令行中的 `--output-dir` 和 `--module-dir` 参数仍具有最高优先级。

## 扩展模块

CMake 配置命令使用 `-B build/release`，并按精度把模块写入
`build/release/extensions/fp32/`、`fp64/`、`dd/`、`dump-fp32/` 或
`dump-fp64/`。不同精度的同名模块不会互相覆盖。

迁移期间，读取模块时会先查找新 `build/release/extensions/` 目录，再查找旧
`tmp/` 和源码根目录。
旧日志及 `run_metadata.json` 中的绝对路径是历史运行信息，无需重写。所有新生成的
图像统一写入 `scratch/`，即使命令行或脚本配置给出了其他图像路径也会重定向到该目录。

临界减速日志按数值后端和扫描变量分开保存在 `somega-fp64/`、`somega-dd/`
和 `sr-fp64/`，防止宽泛的文件通配符把不同精度的数据混在同一次拟合中。

两层旧 `tmp/` 均已完成迁移：原始扫描进入 `runs/`，可展示图表进入
`results/`，旧编译产物进入 `build/archive/`，PDF 预览、缓存、基准和诊断实验进入
`scratch/`。原 `phase_analysis/final` 的成果已提升到 `results/phase-behavior/`，
仍可复用的小工具则放在 `scripts/` 下。
