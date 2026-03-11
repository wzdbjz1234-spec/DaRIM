# DaRIM 项目说明文档

## 1. 项目简介

本项目用于复现并扩展论文 **DaRIM: Data-Driven Adaptive Robust Influence Maximization** 的实验流程，核心目标是在**扩散概率未知且存在估计误差**的情况下，利用数据驱动方式构造边级不确定区间，并在此基础上完成鲁棒影响力最大化（Robust Influence Maximization, RIM）。

项目当前包含两类核心代码：

1. **数据集构造部分**：从真实网络拓扑出发，为节点和边生成特征，并基于超参数模型生成边传播概率，最终保存为带属性的 GraphML 数据集。
2. **主实验部分**：基于生成好的图数据，完成传播样本生成、点估计、bootstrap 重采样、区间构造、LuGreedy 鲁棒选种，以及与 global-δ 基线的对比评估。

整个代码已经从原始的单文件脚本整理为模块化结构，便于后续维护、调参和替换组件。

---

## 2. 项目目标

这个项目主要解决三个问题：

- **如何从网络结构和边特征中生成可控的传播概率数据集**；
- **如何在样本有限的条件下学习扩散模型参数，并量化不确定性**；
- **如何把统计意义上的不确定区间接入鲁棒影响力最大化算法中**。

对应到代码层面，可以理解为三条主线：

- **图数据构造**：原始网络 → 节点特征 → 边特征 → 真实传播概率；
- **统计估计**：传播样本 → 点估计模型 → bootstrap 模型分布 → 边概率区间；
- **鲁棒优化**：区间下界/上界 → LuGreedy → robust ratio 评估 → 与 global-δ 比较。

---

## 3. 项目目录结构

当前推荐的目录结构如下：

```text
project/
├── README_项目说明.md
├── config_refactored.py
├── build_graph_dataset_refactored.py
├── darim_split/
│   ├── __init__.py
│   ├── common.py
│   ├── estimation.py
│   ├── graph_ops.py
│   ├── intervals.py
│   ├── pipeline.py
│   └── main.py
├── data/
│   ├── NetHEPT.txt
│   └── *.graphml
└── result/
    ├── *.csv
    └── *.png
```

如果你当前工程里还保留了 `darim_refactored.py`，可以把它视为**拆分前的单文件版本**；而 `darim_split/` 则是**拆分后的模块化版本**。

---

## 4. 环境依赖

建议使用 Python 3.10 及以上版本。

主要依赖如下：

```bash
pip install numpy scipy networkx matplotlib pandas tqdm mpi4py
```

如果需要并行运行实验，还需要具备 MPI 环境，例如：

- OpenMPI
- MPICH

验证方式示例：

```bash
mpiexec --version
```

---

## 5. 数据集构造模块

### 5.1 作用

数据集构造部分负责把一个原始边表文件（如 `NetHEPT.txt`）转成实验可直接使用的 GraphML 图文件。生成后的图中，每条边通常包含两个关键属性：

- `feature`：边特征向量；
- `probability`：真实传播概率。

这部分代码主要由以下两个文件组成：

- `config_refactored.py`
- `build_graph_dataset_refactored.py`

### 5.2 config_refactored.py

该文件用于统一管理维度相关配置，核心包括：

- `feature_dimension`：边特征维度；
- `theta_dimension`：超参数向量维度；
- `node_embedding_dimension`：节点特征维度。

其中默认假设为：

```text
edge_feature = source_node_feature ⊕ target_node_feature
```

因此通常要求：

```text
feature_dimension = 2 * node_embedding_dimension
```

如果维度不一致，配置类会主动报错，避免后续训练时出现隐藏 bug。

### 5.3 build_graph_dataset_refactored.py

该脚本完成完整的图数据生成流程：

1. 读取原始边表；
2. 为每个节点随机生成特征；
3. 将源节点和目标节点特征拼接为边特征；
4. 采样真实超参数 `theta_true`；
5. 用 `sigmoid(theta · x + bias)` 生成边传播概率；
6. 保存为 GraphML。

### 5.4 常用运行方式

示例：

```bash
python build_graph_dataset_refactored.py \
  --input-path data/NetHEPT.txt \
  --output-path data/nethept.graphml \
  --seed 114514 \
  --theta-low -1.0 \
  --theta-high 0.0
```

若原始图本质上是无向图，希望补成双向边，可增加：

```bash
--add-reverse-edges
```

### 5.5 输出说明

生成后的 GraphML 文件中，每条边一般包含：

- `feature`：字符串形式保存的向量；
- `probability`：浮点数形式的真实扩散概率。

这些属性会在主实验脚本中自动解析回来。

---

## 6. 主实验模块整体说明

主实验模块位于 `darim_split/` 目录下，是从原来的 `darim_refactored.py` 中拆分出来的。

它的整体逻辑可以概括为：

```text
读取图
→ 初始化真实概率映射
→ 生成传播样本
→ 训练点估计模型
→ 训练 bootstrap 模型
→ 构造边概率区间
→ 运行 LuGreedy
→ 评估 robust ratio
→ 与 global-δ 基线比较
→ 保存结果
```

为了让代码更易维护，现在按职责拆成了 6 个模块。

---

## 7. darim_split/ 各模块作用说明

### 7.1 common.py

该文件放置**公共配置、数据结构和通用工具**。

主要内容包括：

- `ExperimentConfig`：实验配置类；
- `FittedModelState`：保存拟合后的 `theta` 和 `bias`；
- `EvaluationRecord`：保存单次实验结果；
- MPI 全局对象：`comm / rank / size`；
- `load_graph()`：读取 GraphML 或 txt 图文件；
- `compute_prob()`：根据特征与参数计算传播概率；
- `GraphAttributeProxy`：统一访问图上的边属性；
- `save_results()`：保存 CSV 和绘制结果图。

这个文件相当于整个项目的“底座”。

### 7.2 estimation.py

该文件负责**样本生成和模型拟合**。

主要内容包括：

- `_generate_sample_batch()`：在 IC 传播逻辑下生成训练样本；
- `mpi_generate_propagation_samples()`：MPI 并行采样入口；
- `HyperparametricModel`：超参数扩散模型；
- `fit_point_estimator()`：拟合点估计模型；
- `fit_bootstrap_models()`：训练多个 bootstrap 模型；
- `precompute_bootstrap_intervals()`：预计算不同 `alpha` 下的边概率区间。

简单理解：这个模块负责“**数据怎么来，参数怎么学，区间怎么准备**”。

### 7.3 graph_ops.py

该文件负责**图扩散、RIS、LuGreedy 和 robust ratio 评估**。

主要内容包括：

- RR sets 生成；
- 分布式 RIS greedy；
- `lugreedy()`：鲁棒选种核心算法；
- `compute_influence_from_seed()`：根据给定 seedset 估计影响力；
- `evaluate_robust_ratio()`：评估 seedset 的鲁棒比值；
- `initialize_ground_truth_probabilities()`：初始化真实概率映射。

简单理解：这个模块负责“**给定不确定区间后，怎么选种，怎么评估**”。

### 7.4 intervals.py

该文件负责**区间写入、覆盖率/宽度计算、以及 global-δ 的匹配**。

主要内容包括：

- `write_interval_attributes_for_alpha()`：把某个 `alpha` 对应的 bootstrap 区间写回图边属性；
- `cleanup_interval_attributes()`：清理临时区间属性；
- `compute_interval_coverage_proxy()`：计算区间覆盖率；
- `compute_avg_width_proxy()`：计算平均区间宽度；
- `update_graph_delta_intervals()`：生成 global-δ 区间；
- `find_delta_for_target_coverage()`：找到覆盖率匹配的 `delta`；
- `find_delta_for_target_width()`：找到宽度匹配的 `delta`。

简单理解：这个模块负责“**把 bootstrap 区间和基线 δ 都变成可以比较的形式**”。

### 7.5 pipeline.py

该文件负责**实验主流程编排**。

主要内容包括：

- `run_global_delta_baseline()`：给定 `delta`，运行一次 global-δ 基线；
- `run_darim_pipeline()`：遍历全部 `alpha`，依次完成：
  - bootstrap 区间写入；
  - LuGreedy 选种；
  - robust ratio 评估；
  - 与 coverage 匹配基线比较；
  - 与 width 匹配基线比较；
  - 记录结果。

简单理解：这个模块负责“**组织整场实验怎么跑**”。

### 7.6 main.py

该文件是**程序入口**，只负责把前面几个模块串起来。

执行顺序通常是：

1. 构建配置；
2. 读取图；
3. 初始化真实概率；
4. 生成传播样本；
5. 拟合点估计模型；
6. 拟合 bootstrap 模型；
7. 调用 `run_darim_pipeline()`；
8. 保存结果。

简单理解：`main.py` 只负责调度，不负责具体算法细节。

---

## 8. 实验配置说明

实验参数集中定义在 `darim_split/common.py` 的 `ExperimentConfig` 中。

默认参数包括但不限于：

- `theta_dimension`：模型维度；
- `global_seed`：随机种子；
- `num_epochs`：模型训练轮数；
- `bootstrap_num`：bootstrap 重采样次数；
- `data_size`：传播样本总量；
- `rrset_num`：RR sets 数量；
- `fixed_seed_size`：选种预算；
- `influence_simul`：影响力 Monte Carlo 模拟次数；
- `alpha_list`：不同置信区间水平对应的实验列表；
- `graph_path`：输入图路径；
- `output_csv` / `output_plot_png`：输出结果路径。

如果要修改实验规模，优先改这里。

---

## 9. 主实验运行方式

### 9.1 单进程运行

```bash
python -m darim_split.main
```

### 9.2 MPI 并行运行

例如使用 4 个进程：

```bash
mpiexec -n 4 python -m darim_split.main
```

### 9.3 运行前需要确认

请先检查：

- `ExperimentConfig.graph_path` 是否指向正确的 GraphML 文件；
- 图中每条边是否包含 `feature` 和 `probability` 属性；
- `theta_dimension` 是否和边特征维度一致；
- `result/` 输出目录是否有写权限。

---

## 10. 输入与输出

### 10.1 输入

主实验通常输入的是一个 GraphML 图文件，其中边上至少要有：

- `feature`
- `probability`

### 10.2 输出

实验结果一般包括两部分：

1. **CSV 文件**：记录每个 `alpha` 下的实验统计量；
2. **PNG 图像**：展示 bootstrap、risk-match、width-match 的对比曲线。

CSV 常见字段包括：

- `alpha`
- `boot_ratio`
- `risk_ratio`
- `width_ratio`
- `risk_consistency`
- `width_consistency`
- `boot_coverage`
- `boot_width`
- `delta_risk`
- `delta_width`

---

## 11. 一次完整实验是怎么跑起来的

可以把整套流程理解成下面这样：

### 第一步：构造图数据

使用 `build_graph_dataset_refactored.py` 从原始边表生成带特征和传播概率的 GraphML。

### 第二步：读取图并生成传播样本

`main.py` 读取图后，调用 `mpi_generate_propagation_samples()` 生成训练样本，每条样本表示某个节点在一组激活父节点影响下是否被成功激活。

### 第三步：拟合点估计模型

使用 `fit_point_estimator()` 训练一个超参数模型，得到点估计概率 `point_prob_map`。

### 第四步：拟合 bootstrap 模型分布

使用 `fit_bootstrap_models()` 对训练样本做有放回重采样，并重复训练多个模型，得到一组 `(theta, bias)`。

### 第五步：构造 bootstrap 区间

使用 `precompute_bootstrap_intervals()` 将一组模型状态投影到各条边上，并为每个 `alpha` 生成边级区间。

### 第六步：运行鲁棒优化

对于每个 `alpha`：

- 将区间写入边属性；
- 在区间上下界下运行 `lugreedy()`；
- 得到 bootstrap 方法下的 seedset 和 robust ratio。

### 第七步：构造并比较 global-δ 基线

针对 bootstrap 区间的覆盖率和宽度，分别寻找能匹配的 `delta`，然后运行两组基线：

- risk-match baseline
- width-match baseline

### 第八步：保存结果

最终把每个 `alpha` 的对比结果写入 CSV，并绘制结果图。

---

## 12. 这次重构的意义

这次整理工作的重点不是“改变论文方法”，而是“把代码职责拆清楚”。

相比原始单文件脚本，现在的好处是：

- **更容易定位问题**：采样、拟合、区间、优化彼此分开；
- **更容易调参**：大部分实验参数集中在 `ExperimentConfig`；
- **更容易替换模块**：例如以后可以单独替换估计器，或者更换基线算法；
- **更适合长期维护**：以后长时间不看代码，回来也能快速找到入口。

---

## 13. 后续建议

如果后续继续维护这个项目，建议按下面方向迭代：

1. **给 `ExperimentConfig` 增加命令行参数支持**，避免每次手改源码；
2. **补充日志系统**，区分信息输出、调试输出和错误输出；
3. **增加单元测试**，尤其是：
   - 样本生成逻辑；
   - interval 写入逻辑；
   - LuGreedy 输出一致性；
   - robust ratio 评估函数；
4. **统一数据路径管理**，把 `data/` 和 `result/` 目录规范固定下来；
5. **把实验脚本和论文作图脚本进一步分离**，减少主流程文件职责。

---

## 14. 快速上手建议

如果你隔了很久再回来继续做实验，建议按下面顺序读代码：

1. 先看 `README_项目说明.md`；
2. 再看 `darim_split/main.py`，理解主流程入口；
3. 再看 `pipeline.py`，理解实验循环；
4. 再分别看：
   - `estimation.py`
   - `graph_ops.py`
   - `intervals.py`
5. 最后再看 `build_graph_dataset_refactored.py`，回顾数据是怎么造出来的。

这样重新进入状态会最快。

---

## 15. 备注

本项目目前默认围绕 **Independent Cascade (IC)** 扩散模型、**超参数扩散概率建模**、**bootstrap 不确定性量化** 和 **LuGreedy 鲁棒选种** 展开。

如果后续需要扩展到：

- 不同扩散模型；
- 其他参数估计器；
- 其他鲁棒优化基线；
- 更多数据集与实验设置；

建议优先保持当前模块边界不变，再在局部模块中扩展实现。

---

## 16. 一句话总结

这个项目现在的结构可以概括为：

**`build_graph_dataset_refactored.py` 负责造图，`darim_split/main.py` 负责跑实验，`darim_split/` 下面的各模块分别负责采样、估计、区间构造和鲁棒优化。**

