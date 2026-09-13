# 项目路径约定

所有活跃脚本的默认路径均以 Git 仓库根目录为基准，不再依赖运行命令时的当前目录。

| 目录 | 用途 | 是否进入 Git |
|---|---|---|
| `build/` | Python 扩展、链接库、可执行文件、性能分析文件 | 否 |
| `runs/` | 原始扫描、轨迹、日志和可再生成数据 | 否 |
| `results/` | 报告引用的图片、汇总表和精选结果 | 是 |
| `scratch/` | smoke test、一次性诊断和临时文件 | 否 |
| `tmp/` | 尚未完成归档的历史数据，只用于兼容读取 | 否 |

## 可配置目录

可以设置 `IYPT16_BUILD_DIR`、`IYPT16_RUNS_DIR`、`IYPT16_RESULTS_DIR` 或
`IYPT16_SCRATCH_DIR`，把大数据放到其他磁盘而无需修改源码。命令行中的
`--output-dir` 和 `--module-dir` 参数仍具有最高优先级。

## 扩展模块

`build.py` 按精度写入 `build/extensions/fp32/`、`fp64/` 或 `dd/`；
`build_dump.py` 写入 `dump-fp32/` 或 `dump-fp64/`。不同精度的同名模块
不会再互相覆盖。

迁移期间，读取模块时会先查找新 `build/` 目录，再查找旧 `tmp/` 和源码根目录。
旧日志及 `run_metadata.json` 中的绝对路径是历史运行信息，无需重写。

临界减速日志按数值后端和扫描变量分开保存在 `somega-fp64/`、`somega-dd/`
和 `sr-fp64/`，防止宽泛的文件通配符把不同精度的数据混在同一次拟合中。
