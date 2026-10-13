# rag-ds-evaluator

用于研究 claim-level RAG 证据状态诊断的 Python 项目。

当前进度：总规划的 14 个生成步骤已完成 1–14，**RefChecker NLI checker 的
真实输出已接入并在 v1 / v2 两个数据集上跑出结果**（导出输入 → 推理 →
摄取 → 谱系记录 → 阈值搜索 → 显著性检验 → 嵌套交叉验证）。
适配器层（RAGChecker 全流程 / RAGAS / LLM）仍然只有**接口与转换**，不导入也
不调用这些第三方库；RAGAS 接入与 SURE-RAG 尚未开展。多评估器实验已在 v2 上
完成（两个 NLI 模型 + 一个生成式 flan-t5 评判器），K_eval 有真实取值；
结论是多评估器融合**没有**优于最好的单评估器，详见「二、多评估器融合为什么
没有增益」。

**2026-09-29 至 10-02 补充的对照改写了主结论的归因**：判定规则与 D-S 完全
相同、只把两个坐标轴换成简单统计量的 Statistic Gate（四个变体）与 D-S **打平**，
单 / 双 / 三评估器、估计可靠性、5 个分折种子下都如此；conflict_aware 三个参数
都调好后，D-S 对它的优势也**不再显著**；m_theta / K_doc 作为检测分数与简单
统计量**打平**；v2 嵌套消融则确认两个判定维度**各自显著有用**。把 NLI 输入换成
人工投票（上限分析）后，两维判定显著优于调充分的 conflict_aware（+0.10 ~
+0.13），但起作用的是「证据不足一维用连乘」，不是 Dempster 规则。D-S 相对
朴素 baseline 的优势应归功于「证据不足 + 文档冲突」两维判定框架，而不是
Dempster 融合本身；当前瓶颈在关系评估器。详见「四」到「七」。

**2026-10-11 至 10-12 补了两个新数据集**：AVeriTeC（标签由事实核查员人工判定，
920 条）上，控制住「证据篇数」这个混杂因素之后，三条结论与 CLIMATE-FEVER 完全
一致；不控制时冲突一维的成绩主要来自「冲突类 claim 配了更多证据」，是假象。
CONFACT（文档来源可信度确有差异，287 条）上，文档级可靠性折扣方向为正但不
显著，且与简单加权打平 —— 不过那里 NLI 的倾向判断接近随机，只算弱证据。
详见「八」「九」。

**2026-10-12 至 10-13 把关系评估器换成大模型（DeepSeek V4.1 Flash），部分结论
随之改变**：下游一个字不改，D-S 的 Macro-F1 从 0.35 / 0.39 升到 0.51 / 0.58
（CLIMATE-FEVER / AVeriTeC），证实瓶颈在关系评估器；两维判定对调充分的
conflict_aware 的优势从不显著变成显著（+0.19 / +0.10，5 个种子全部显著，AVeriTeC
控制证据篇数后仍显著）；D-S 仍不优于同框架里最好的简单统计量（`max_product`
略高）。同一个大模型零样本直接判四类是 0.46 / 0.59：在 CLIMATE-FEVER 上显著低于
两维判定，在 AVeriTeC 上与它打平，并且很少判「冲突」（召回率约 0.20）。只有一个
模型、一份提示词、各跑一遍。详见「十」。

第一版正式数据 `data/processed/climate_fever_v1/` 已从 CLIMATE-FEVER 构建完成
（400 条、四类各 100、train/validation/test = 240/80/80）；v2 扩到 616 条
（四类各 154，train/validation/test = 368/124/124）。两个数据集自带的
`*_relations.jsonl` 都是人工证据投票的 **annotation oracle**，只用于验证
D-S 融合链路，**不能**当作模型实验结果写入论文；模型结果以
`outputs/ragchecker*/` 下的谱系清单为准。

## 环境要求

- Python 3.11（目标版本）；当前开发机未安装 3.11，实际使用 3.12。
- `pyproject.toml` 中约束为 `>=3.11,<3.13`，即 3.11 与 3.12 均可。
- 请勿使用 Python 3.13/3.14：后续接入的 RAGChecker / RAGAS 及其依赖尚无稳定支持。

## 创建虚拟环境

在项目根目录运行（用 py 启动器显式指定版本）：

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
```

若已安装 Python 3.11，可改用 `py -3.11 -m venv .venv`。

## 安装依赖和项目

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

要**逐版本精确复原**论文结果所用的环境，改用锁定文件：

```powershell
python -m pip install -r requirements-main.lock.txt
python -m pip install -e . --no-deps
```

跑 NLI / flan-t5 推理的隔离环境另有 `requirements-refchecker.lock.txt`
（torch 为 CPU 版，所需的 PyTorch 索引已写在文件里）。两个环境的 numpy、
scikit-learn 版本不同，这正是要把它们分开的原因。

仓库里已包含 v2 数据集、全部模型关系文件与实验结果，**不重跑推理也能复现
全部主表**；只有想换模型或换数据时才需要 refchecker 环境。

## 检查项目

```powershell
python -c "import rag_ds; print(rag_ds.__version__)"
python -m pytest -q
```

## 数据模型

所有模型定义在 `src/rag_ds/schemas.py`，基于 Pydantic v2，后续的 RAGChecker、
RAGAS 与 D-S 融合模块共用这一套契约。

| 模型 | 作用 |
| --- | --- |
| `EvidenceState` | 字符串枚举：`supported` / `refuted` / `insufficient` / `conflicting` |
| `Claim` | 从答案中拆分出的一条原子断言（`claim_id`、`text`） |
| `ContextChunk` | 一段检索文档（`doc_id`、`text`、`retrieval_score`、`reliability`） |
| `RAGSample` | 一条完整评估样本：问题、答案、claims、contexts、`gold_state` |
| `RelationPrediction` | 评估器对单个 (claim, document) 对的三元概率输出 |

统一约束：

- 必填字符串自动去首尾空白，去空白后不得为空；
- 所有概率与可靠度字段取值范围为 `[0, 1]`；
- `RAGSample` 内 `claim_id` 与 `doc_id` 必须唯一；
- `RelationPrediction` 的 `p_support + p_refute + p_unknown` 必须为 1，
  允许 `PROBABILITY_SUM_TOLERANCE`（`1e-6`）误差；
- **所有模型均禁止未定义字段**（`extra="forbid"`），拼写错误会立即报错。

## 样例数据

`data/samples/demo.jsonl` 为 UTF-8 编码，每行一条完整的 `RAGSample`，
四条样例分别覆盖四种 `gold_state`：

| sample_id | gold_state | 说明 |
| --- | --- | --- |
| `demo-001` | `supported` | 两段文档共同支持答案中的两条 claim |
| `demo-002` | `refuted` | 文档明确反驳答案 |
| `demo-003` | `insufficient` | 文档未涉及问题所需信息 |
| `demo-004` | `conflicting` | 两段文档结论相反 |

## 正式数据集（CLIMATE-FEVER v1）

`data/processed/climate_fever_v1/` 由官方 CLIMATE-FEVER JSONL 转换而来，不是
模型生成、也不是手工编造的 claim。原始文件保存在
`data/raw/climate-fever.jsonl`，SHA-256 写在 `manifest.json` 里。

| 项 | 取值 |
| --- | --- |
| 来源 | [CLIMATE-FEVER](https://github.com/tdiggelm/climate-fever-dataset)（Diggelmann et al., 2020） |
| 原始规模 | 1,535 条互联网气候 claim，每条 5 段 Wikipedia 证据 |
| 本仓库子集 | 四类各 100 条，共 400 条；seed = 42 |
| 划分 | 每类 60 / 20 / 20 → train 240、validation 80、test 80，分层且互斥 |
| 标签映射 | `SUPPORTS→supported`，`REFUTES→refuted`，`NOT_ENOUGH_INFO→insufficient`，`DISPUTED→conflicting` |
| 关系文件 | `*_relations.jsonl`，评估器名 `climate_fever_human_vote_distribution` |
| 谱系 | 每个 split 一份 `*_provenance.jsonl`，含原始 claim_id、Wikipedia 条目与投票 |

**关系概率是人工标注 oracle。** 它们由证据投票分布换算而来，用于检查折扣、
融合、K_doc / K_eval 与二维门控是否正确；不能报告为 RAGChecker、RAGAS 或
任何 LLM 评估器的性能。官方仓库没有单独的数据集许可证文件，学术使用须引用
原论文，并遵守底层英文 Wikipedia 内容的许可与署名要求。

重建（已存在时需 `--overwrite`）：

```powershell
python scripts/prepare_climate_fever.py --overwrite
```

在验证集上搜索四分类阈值（只搜 `theta` × `K_doc` 共 25 点；`K_eval` 是固定
告警阈值，不参与 Macro-F1 网格）：

```powershell
python scripts/tune_thresholds.py `
  --manifest data/processed/climate_fever_v1/manifest.json `
  --samples data/processed/climate_fever_v1/validation.jsonl `
  --predictions data/processed/climate_fever_v1/validation_relations.jsonl `
  --out outputs/metrics/climate_fever_threshold_search.json
```

调参脚本会核对 manifest、路径、记录数和 SHA-256；拿训练集、测试集或被改过
的验证集调参会直接失败。把验证集上选出的阈值手工填入
`configs/climate_fever_oracle_test.yaml` 后再跑测试集。当前该配置里的
`theta_threshold: 0.3`、`document_conflict_threshold: 0.6` 来自这次验证集
搜索；`evaluator_conflict_threshold: 0.4` 未参与搜索。

## 数据读写

读写函数定义在 `src/rag_ds/data_io.py`，只做序列化与校验，不含任何算法。

| 函数 | 作用 |
| --- | --- |
| `iter_samples(path)` | 流式逐行读取，产出校验通过的 `RAGSample` 迭代器 |
| `load_samples(path)` | 一次性读取为列表，内部复用 `iter_samples` |
| `write_samples(path, samples, overwrite=False)` | 写出 JSONL，返回写入条数 |
| `iter_relation_predictions(path)` | 同上，但产出 `RelationPrediction` |
| `load_relation_predictions(path)` | 同上，一次性读为列表 |
| `write_relation_predictions(path, predictions, overwrite=False)` | 同上，写出关系预测 |

行为约定：

- 读取使用 `utf-8-sig`，兼容带 BOM 的文件；纯空白行被跳过，但行号照常累加；
- 文件不存在时在调用瞬间抛出 `FileNotFoundError`，不会拖到迭代时才报错；
- 某行不是合法 JSON、不是 JSON 对象、或不通过 `RAGSample` 校验时，
  抛出 `JsonlDataError`，信息中包含文件路径、物理行号与简要原因，
  并以 `.path` / `.line_number` / `.reason` 属性暴露，便于程序化处理；
  错误信息不会打印整份文件；
- 写入先落到同目录临时文件，成功后用 `os.replace` 原子替换，
  失败则清理临时文件并保留原文件；
- 中文原样保留（`ensure_ascii=False`），不会转义成 Unicode 码点形式；
  换行统一为 LF，文件末尾保留换行符；
- 目标文件已存在时默认拒绝写入，需显式传 `overwrite=True`。

用法示例：

```python
from rag_ds import load_samples, write_samples

samples = load_samples("data/samples/demo.jsonl")
write_samples("outputs/predictions/subset.jsonl", samples[:2], overwrite=True)
```

## 关系评估器

接口与实现在 `src/rag_ds/relation_evaluation/`。

- **`RelationEvaluator`**（`base.py`）—— 抽象基类。`name` 属性给出评估器名称；
  `evaluate(sample, claim, context)` 判断单个组合；`evaluate_sample(sample)`
  按「外层 claims、内层 contexts」的固定顺序遍历全部组合。基类只定义接口与
  遍历顺序，不含任何判断规则。
- **`MockRelationEvaluator`**（`mock.py`）—— 查表式假评估器。构造时按
  `(evaluator, sample_id, claim_id, doc_id)` 建索引，只装载 `evaluator` 字段
  与自身 `name` 相同的记录；重复键立即报 `ValueError`。查不到时抛
  `MissingMockPredictionError`，信息含全部四个 ID。返回值为深拷贝。
- 它**不读取** `gold_state`，也不读取 `question` / `answer` / `claim.text` /
  `context.text`。`gold_state` 是实验标签，用它生成预测会造成数据泄漏。

预设数据 `data/samples/mock_relations.jsonl` 共 5 行，ID 与 `demo.jsonl` 对应：

| sample_id | claim / doc | p_support | p_refute | p_unknown |
| --- | --- | --- | --- | --- |
| `demo-001` | `c1` / `d1` | 0.90 | 0.05 | 0.05 |
| `demo-002` | `c1` / `d1` | 0.05 | 0.90 | 0.05 |
| `demo-003` | `c1` / `d1` | 0.05 | 0.05 | 0.90 |
| `demo-004` | `c1` / `d1` | 0.90 | 0.05 | 0.05 |
| `demo-004` | `c1` / `d2` | 0.05 | 0.90 | 0.05 |

注意 `demo-001` 有 2 个 claim 和 2 段文档（共 4 个组合），预设只覆盖了第一个组合，
因此对该样本调用 `evaluate_sample` 会抛 `MissingMockPredictionError`。
这是刻意保留的缺失用例。

用法示例：

```python
from rag_ds import MockRelationEvaluator, load_relation_predictions, load_samples

presets = load_relation_predictions("data/samples/mock_relations.jsonl")
evaluator = MockRelationEvaluator("mock_evaluator", presets)

sample = next(s for s in load_samples("data/samples/demo.jsonl") if s.sample_id == "demo-004")
for prediction in evaluator.evaluate_sample(sample):
    print(prediction.doc_id, prediction.p_support, prediction.p_refute)
```

## BPA 映射与可靠性折扣

代码在 `src/rag_ds/ds/`。识别框架固定为两个互斥假设：

```
Theta = {Support, Refute}
```

幂集上只有三个焦元可以承载质量：`{Support}`、`{Refute}` 和 `Theta` 本身。

### m_theta 的含义

`m_theta` 是分配给**整个识别框架**的质量，表示「当前证据尚不能区分支持与反驳」，
即无知。它**不是**与 Support、Refute 并列的第三个互斥类别 —— 这个区别在后续
Dempster 组合中会体现出来：Theta 上的质量可以与任一焦元相交并让渡给对方，
而三个互斥类别之间只会产生冲突。

### 基础 BPA 映射（`ds/mass.py`）

`mass_from_prediction(prediction) -> MassFunction`：

```
m(S)     = p_support
m(R)     = p_refute
m(Theta) = p_unknown
reliability_applied = 1.0
```

这一步**不应用任何可靠性**，`evaluator_reliability` 被刻意忽略。

`MassFunction` 是 Pydantic v2 的不可变模型（`frozen=True`），字段为
`sample_id`、`claim_id`、`doc_id`、`evaluator`、`m_support`、`m_refute`、
`m_theta`、`reliability_applied`。三个质量各自位于 [0, 1] 且和为 1
（容差 `MASS_SUM_TOLERANCE`，与 `PROBABILITY_SUM_TOLERANCE` 取同一个值）。

### 可靠性折扣（`ds/discount.py`）

`discount_mass(mass, reliability) -> MassFunction`：

```
m'(S)     = r * m(S)
m'(R)     = r * m(R)
m'(Theta) = 1 - m'(S) - m'(R)
reliability_applied' = reliability_applied * r
```

质量只从确定焦元流向 Theta，绝不反向流动。`r = 1` 时数值不变；`r = 0` 时
退化为 `m_theta = 1` 的完全无知。连续折扣等价于乘积折扣：
`discount(discount(m, a), b)` 与 `discount(m, a * b)` 结果相同。

`discounted_mass_from_prediction(prediction, context)` 是完整链路：先校验两个
`doc_id` 一致（不一致抛 `ValueError`，信息含两个 ID），再取

```
r_effective = context.reliability * prediction.evaluator_reliability
```

**`retrieval_score` 不参与可靠性计算。** 检索相关性衡量「这段文档与问题有多相关」，
与「这段文档有多可信」是两回事，混用会让折扣失去意义。

### 数值示例

输入 `p = (0.8, 0.1, 0.1)`，文档可靠性 0.9，评估器可靠性 0.8：

| 量 | 值 |
| --- | --- |
| `r_effective` | 0.9 × 0.8 = 0.72 |
| `m(S)` | 0.72 × 0.8 = 0.576 |
| `m(R)` | 0.72 × 0.1 = 0.072 |
| `m(Theta)` | 1 − 0.576 − 0.072 = 0.352 |

浮点实际值为 0.5760000000000001 / 0.07200000000000001 / 0.3519999999999999，
因此测试一律使用 `pytest.approx`，不做直接相等比较。

用法示例：

```python
from rag_ds import ContextChunk, RelationPrediction, discounted_mass_from_prediction

prediction = RelationPrediction(
    sample_id="s1", claim_id="c1", doc_id="d1", evaluator="mock_evaluator",
    p_support=0.8, p_refute=0.1, p_unknown=0.1, evaluator_reliability=0.8,
)
context = ContextChunk(doc_id="d1", text="文档正文。", reliability=0.9)

mass = discounted_mass_from_prediction(prediction, context)
print(mass.m_support, mass.m_refute, mass.m_theta)
```

## 两条 BPA 的 Dempster 组合

代码在 `src/rag_ds/ds/combination.py`。只实现**标准归一化 Dempster 规则**，
不含 Yager、Dubois-Prade 或任何未归一化变体。

### 公式

单次冲突量，在归一化**之前**计算：

```
K = m1(S) * m2(R) + m1(R) * m2(S)
```

未归一化质量：

```
S_raw     = m1(S)m2(S) + m1(S)m2(Theta) + m1(Theta)m2(S)
R_raw     = m1(R)m2(R) + m1(R)m2(Theta) + m1(Theta)m2(R)
Theta_raw = m1(Theta)m2(Theta)
```

归一化，分母为 `1 - K`：

```
m(S)     = S_raw / (1 - K)
m(R)     = R_raw / (1 - K)
m(Theta) = Theta_raw / (1 - K)
```

### 为什么 K 必须单独保留

K 是归一化之前被两条证据判定为互相矛盾的那部分质量。归一化把它从分子中抹掉、
再把剩余质量放大回和为 1，因此**融合结果本身无法反映原始冲突有多大**：两条温和
一致的证据与两条剧烈矛盾的证据完全可能给出相近的融合质量（Zadeh 反例的根源）。
所以 `PairwiseCombinationResult` 把 K 与融合结果一起保存 —— 丢掉 K 就等于丢掉
「这个结论有多可疑」这一信息。

### K 的命名边界

本模块的 `conflict` 只是**两条 BPA 的单次冲突**，既不是 `K_doc` 也不是 `K_eval`。
后者是聚合层按证据来源（同一评估器下的多篇文档 / 同一文档下的多个评估器）分别
累计出来的量，将在后续阶段实现。在这里叫它 K_doc 会把两个层次的量混为一谈。

### 数据模型

- **`CombinedMass`** —— 只有 `m_support`、`m_refute`、`m_theta` 三个字段。
  **刻意不携带任何 ID**：融合结果来自两条证据，任何单一 ID 都是伪造的，
  而 `"doc1+doc2"`、`"combined"` 这类拼接值会让下游误以为它是一篇真实文档。
  融合来源与业务元数据由后续聚合层单独记录。
- **`PairwiseCombinationResult`** —— `mass` / `conflict` / `normalization_denominator`，
  并校验 `normalization_denominator == 1 - conflict`。
- 两个模型都是 `frozen=True` 的不可变模型。

### 完全冲突

`1 - K <= TOTAL_CONFLICT_EPSILON`（1e-12）时抛出 `TotalConflictError`，
异常携带 K 与 1-K 的实际数值。本项目**不会**用 epsilon 替代分母强行计算、
不返回全零质量、不返回 `m_theta = 1`、不返回任一侧证据 —— 这些做法都会
悄悄改变算法含义。

数值提醒：`1 - K` 由两个接近 1 的数相减得到，K 逼近 1 时发生灾难性抵消。
实测在 `1 - K` 处于约 `[1e-12, 2e-11]` 区间时，归一化后三个质量之和偏离 1
已超过 `MASS_SUM_TOLERANCE`，此时 `CombinedMass` 会抛校验错误而不是返回结果。
这是刻意的：分母的有效位数已所剩无几，宁可大声报错也不返回不可信数值。
`1 - K` 大于该区间时归一化稳定。

### 数值示例

输入 `left = (0.8, 0.1, 0.1)`，`right = (0.1, 0.8, 0.1)`：

| 量 | 值 |
| --- | --- |
| `K` | 0.8×0.8 + 0.1×0.1 = 0.65 |
| `1 - K` | 0.35 |
| `S_raw` / `R_raw` / `Theta_raw` | 0.17 / 0.17 / 0.01 |
| `m(S)` / `m(R)` / `m(Theta)` | 0.4857142857 / 0.4857142857 / 0.0285714286 |

浮点实际值分别为 0.485714285714286 与 0.028571428571428588，
因此测试一律使用 `pytest.approx`。

用法示例：

```python
from rag_ds import CombinedMass, TotalConflictError, combine_two_masses

left = CombinedMass(m_support=0.8, m_refute=0.1, m_theta=0.1)
right = CombinedMass(m_support=0.1, m_refute=0.8, m_theta=0.1)

try:
    result = combine_two_masses(left, right)
except TotalConflictError as error:
    print("完全冲突：", error.conflict)
else:
    print(result.mass.m_support, result.conflict)
```

## 完整证据链路

```
关系概率
  -> 文档可靠性折扣        （每条文档一次，用 context.reliability）
  -> 同一评估器内融合文档   -> 评估器级 BPA 与该评估器的 K_doc
  -> 评估器可靠性折扣      （每个评估器只有一次，用 evaluator_reliability）
  -> 融合多个评估器        -> 最终 BPA、K_eval 与加权 K_doc
```

**评估器可靠性必须在文档融合之后只作用一次。** 若在每条文档上都乘一遍，
它会随文档数量被重复计入：同一个评估器看了 5 篇文档，可靠性就被折了 5 次，
结果凭空受文档数量影响。实测折扣前后 m(S) 的比值在 1/2/3/5 篇文档下恒为
`evaluator_reliability`，而不是它的 n 次方。

### 三个指标互不等价

| 指标 | 含义 |
| --- | --- |
| `m_theta` | 融合后仍未分配给支持或反驳的质量，衡量**无知** —— 没人给出明确意见 |
| `K_doc` | 同一评估器内**文档之间**的冲突 —— 有文档说支持、有文档说反驳 |
| `K_eval` | **评估器之间**的冲突 —— 不同评估器对同一 claim 给出相反结论 |

三者可任意组合出现，不能互相替代。两个评估器可能各自内部毫无冲突
（K_doc = 0）却彼此对立（K_eval 高）；也可能所有证据都很弱（m_theta 高）
而谁也不与谁矛盾（两个 K 都为 0）。

## 多文档融合与 K_doc

代码在 `src/rag_ds/ds/document_aggregation.py`。

输入必须同属一个 `sample_id` / `claim_id` / `evaluator`，但来自不同 `doc_id`，
且**已完成文档可靠性折扣**。以第一条文档为初始累计 BPA，按**输入原始顺序**
依次融合，不按 `retrieval_score` 或 `doc_id` 重排。

```
K_doc = 1 - (1 - K_1)(1 - K_2) ... (1 - K_n)
```

其中 K_i 是累计 BPA 与第 i+1 条文档 BPA 的单次冲突量。例如 K1=0.2、K2=0.3 时
`K_doc = 1 - 0.8 x 0.7 = 0.44`。

- K_doc **不是** K_i 的平均值；各步 K_i 全部保存在 `steps` 里供方法对比；
- K_doc **不等于** `m_theta`，见上表；
- 只有一条文档时不调用 `combine_two_masses`，`K_doc = 0`，`steps` 为空；
- 完全无知的文档不会增加 K_doc。

**空输入抛 `EmptyEvidenceError`**，不返回全无知 BPA ——「一条文档都没检索到」
是检索环节的问题，伪造成 `m_theta=1` 会让「没有证据」和「证据说不清楚」
在下游无法区分。

**完全冲突时** `mass=None`、`k_doc=1`、`is_total_conflict=True`，停止融合后续
文档但保留全部已知文档 ID。不会被伪造成 `m_theta=1`。

## 多评估器融合与 K_eval

代码在 `src/rag_ds/ds/evaluator_aggregation.py`。

```
K_eval = 1 - (1 - K_1)(1 - K_2) ... (1 - K_n)
```

各评估器 K_doc 的可靠性加权汇总：

```
k_doc_weighted = sum(r_e x K_doc,e) / sum(r_e)
```

`sum(r_e) = 0`（所有评估器都完全不可信）时定义为 0，此时最终质量为
`m_theta = 1` 的完全无知。

- 单评估器时 `K_eval = 0`、`steps` 为空，`mass` 就是那一次折扣后的质量；
- 每个评估器**原始的** K_doc 保留在 `evaluator_diagnostics` 中，不是只留加权
  平均值 —— 加权平均把「某一个评估器内部剧烈冲突」和「所有评估器都轻微冲突」
  压成了同一个数字；
- 诊断同时保留评估器折扣**前后**的质量，供后续消融实验使用。

**某个评估器的文档级 BPA 因完全冲突而 `mass=None` 时**，抛出
`UndefinedDocumentMassError`（携带 sample_id、claim_id、evaluator 与该评估器的
K_doc）。不跳过、不替换成全无知或任一侧质量 —— 文档级完全冲突是需要被下游
直接诊断的结论。

> **pipeline 层的例外**：评估器可靠性为 0 时，其折扣后的质量是完全无知
> （Dempster 融合的单位元），对结论毫无贡献。因此 pipeline 只在**可靠性
> 大于 0** 的评估器出现文档级完全冲突时才把整条 claim 诊断为
> `document_total_conflict`；可靠性为 0 且 `mass=None` 的评估器不进入
> 评估器融合（避免触发上面的异常），但其文档级结果仍完整保留在
> `document_results` 中供诊断。若全部评估器都可靠性为 0 且文档级完全
> 冲突，结论为完全无知（`m_theta = 1`，insufficient），不报错也不伪造
> 冲突结论。

### 接口变更（第八阶段）

`discounted_mass_from_prediction` 与 `effective_reliability` **已废弃**，
调用会发出 `DeprecationWarning`：

| 旧接口 | 现在应使用 |
| --- | --- |
| `discounted_mass_from_prediction` | `document_discounted_mass_from_prediction`（只应用 `context.reliability`） |
| `effective_reliability` | 文档级用 `context.reliability`；评估器级把 `evaluator_reliability` 交给 `discount_combined_mass` |

旧函数保留仅为兼容，行为已与新函数一致，**不再重复应用评估器可靠性**。

### 数值示例

`p = (0.8, 0.1, 0.1)`，文档可靠性 0.9，评估器可靠性 0.8：

| 阶段 | m(S) | m(R) | m(Theta) |
| --- | --- | --- | --- |
| 文档折扣后 | 0.72 | 0.09 | 0.19 |
| 评估器折扣后 | 0.576 | 0.072 | 0.352 |

用法示例：

```python
from rag_ds import (
    EvaluatorEvidence,
    aggregate_document_masses,
    aggregate_evaluators,
    document_discounted_mass_from_prediction,
)

document_masses = [
    document_discounted_mass_from_prediction(prediction, contexts[prediction.doc_id])
    for prediction in predictions
]
document_result = aggregate_document_masses(document_masses)

result = aggregate_evaluators(
    [EvaluatorEvidence(document_result=document_result, evaluator_reliability=0.8)]
)
print(result.mass, result.k_eval, result.k_doc_weighted)
```

## 二维门控诊断

代码在 `src/rag_ds/diagnostics/`。门控**只用两个坐标轴**：

```
横轴：m_theta   证据不足程度
纵轴：K_doc     文档冲突程度
```

| m_theta | K_doc | region | primary_state |
| --- | --- | --- | --- |
| 低 | 低 | `sufficient_consistent` | 由 verdict 决定：supported / refuted / None |
| 高 | 低 | `insufficient` | `INSUFFICIENT` |
| 低 | 高 | `document_conflict` | `CONFLICTING` |
| 高 | 高 | `insufficient_and_conflicting` | `CONFLICTING` |

另有两个非二维状态：`document_total_conflict`（文档融合完全冲突）与
`evaluator_total_conflict`（评估器融合完全冲突）。

判定「高」统一使用 `value >= threshold`（含等号），三处一致，不混用 `>` 与 `>=`。

混合区域映射为 `CONFLICTING` 而不是 `INSUFFICIENT`，否则冲突信息会在四分类里
彻底丢失；`evidence_insufficient` 与 `document_conflict` 两个布尔字段仍同时为
`True`，完整信息不丢。

### K_eval 是额外警报，不是坐标轴

`K_eval` **只翻转 `evaluator_disagreement` 这一个布尔字段**，不改变 `region`、
`primary_state`、`evidence_insufficient` 或 `document_conflict`。把 K_eval 从
0.1 改到 0.8，逐字段比对确认只有 `k_eval` 与 `evaluator_disagreement` 变化。

三个诊断量互不替代：

| 量 | 衡量什么 |
| --- | --- |
| `m_theta` | 证据不足（无知）—— 没人给出明确意见 |
| `K_doc` | 文档之间冲突 —— 有文档说支持、有文档说反驳 |
| `K_eval` | 评估器意见冲突 —— 不同评估器给出相反结论 |

### 支持/反驳倾向

`determine_verdict(m_support, m_refute, tie_tolerance)`：

```
margin = m_support - m_refute
margin >  tie_tolerance  ->  supported
margin < -tie_tolerance  ->  refuted
否则                      ->  undetermined
```

`m_theta` 不参与倾向判断；平局一律 `undetermined`，不随机打破，也不默认判为
supported。

verdict 与 region 是两件事：`demo-004` 落在 `document_conflict` 区域，融合质量
却仍偏向 `refuted`，两个信息都被保留。

### 完全冲突不等于完全无知

`m_theta = 1` 表示完全无知（谁也没给出意见）；完全冲突表示证据非常明确、只是
彼此对立到标准 Dempster 规则无法归一化。两者含义不同，因此完全冲突时三个质量
一律为 `None`，**不会**被伪造成 `m_theta = 1`，也不会被写成「证据不足」。

文档完全冲突用专用函数 `diagnose_document_total_conflict`，它只接受
`is_total_conflict=True` 的结果，输出 `primary_state=CONFLICTING`；其他输入抛
`ValueError`。

### 阈值

> **以下取值仅为调试默认值，不是通过任何数据选出的正式阈值。**
> 正式实验必须在**验证集**上选择阈值，**测试集不得参与选择**。
> 本阶段不实现任何阈值搜索或训练。

```yaml
diagnostics:
  theta_threshold: 0.5
  document_conflict_threshold: 0.4
  evaluator_conflict_threshold: 0.4
  tie_tolerance: 0.000001
```

配置写在 `configs/default.yaml`；本阶段只保存配置，不实现自动加载器。
每个 `DiagnosticResult` 都随结果保存本次实际使用的 `thresholds`，便于复现。

### 四类 demo 的实际诊断结果

| sample | m(S) | m(R) | m(Θ) | K_doc | region | verdict | primary_state | gold_state |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| demo-001 | 0.8550 | 0.0475 | 0.0975 | 0.0000 | `sufficient_consistent` | supported | supported | supported |
| demo-002 | 0.0460 | 0.8280 | 0.1260 | 0.0000 | `sufficient_consistent` | refuted | refuted | refuted |
| demo-003 | 0.0425 | 0.0425 | 0.9150 | 0.0000 | `insufficient` | undetermined | insufficient | insufficient |
| demo-004 | 0.4081 | 0.5349 | 0.0570 | 0.6650 | `document_conflict` | refuted | conflicting | conflicting |

四条 `primary_state` 与标注一致。`gold_state` 只在比较时读取，从未进入计算 ——
有一个测试把它抹成 `None` 后重跑，断言诊断结果逐字段不变。

用法示例：

```python
from rag_ds import DiagnosticThresholds, diagnose_evaluator_result

result = diagnose_evaluator_result(evaluator_result, DiagnosticThresholds())
print(result.region, result.verdict, result.primary_state)
print(result.evidence_insufficient, result.document_conflict, result.evaluator_disagreement)
```

## 运行离线 MVP

```bash
python scripts/run_demo.py --config configs/demo.yaml
```

重复运行需要覆盖已有输出时：

```bash
python scripts/run_demo.py --config configs/demo.yaml --overwrite
```

配置里的相对路径**以配置文件所在目录为基准**解析，因此从项目根目录、从
`scripts/` 里、还是从任意别处运行，结果都一样。`--config` 省略时默认使用
项目内的 `configs/demo.yaml`（同样按脚本位置推算，不看终端当前目录）。

### 结果的解释边界

> - 输入的关系概率来自 `data/samples/mock_relations.jsonl` 里**预设的 mock 值**，
>   不是任何模型的真实输出；
> - 当前结果只验证 **D-S 诊断流程本身是否正确**（折扣顺序、融合、K_doc /
>   K_eval、二维门控、输出格式）；
> - 当前结果**不能**说明任何语言模型的评估效果，也不构成任何实验结论；
> - RAGChecker / RAGAS 要到后续阶段才接入。

### 链路

```
RAGSample + RelationPrediction
  -> 文档可靠性折扣      document_discounted_mass_from_prediction
  -> 文档融合与 K_doc    aggregate_document_masses
  -> 评估器可靠性折扣    （aggregate_evaluators 内部只施加一次）
  -> 评估器融合与 K_eval
  -> 二维门控            diagnose_evaluator_result
  -> JSONL / CSV 输出
```

`pipeline.py` **只做编排**：不含任何数学公式，不复制 D-S 计算。

### 输出

| 文件 | 内容 |
| --- | --- |
| `outputs/predictions/demo_diagnostics.jsonl` | 每条 claim 一行，保留完整嵌套的中间过程（各文档 BPA、逐步 K_i、评估器诊断） |
| `outputs/predictions/demo_diagnostics.csv` | 每条 claim 一行的扁平摘要，UTF-8 with BOM，Excel 可直接打开 |

CSV 在完全冲突状态下把三个质量列**留空**，不写 0 也不写 "None" —— 避免下游把
「未定义」误读成「零质量」。多个评估器用 `|` 连接。

### 四条 demo 样本、五条 claim 的实际结果

| claim | 文档数 | m(S) | m(R) | m(Θ) | K_doc | region | verdict | primary_state | gold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| demo-001-c1 | 2 | 0.8557 | 0.0519 | 0.0925 | 0.0406 | `sufficient_consistent` | supported | supported | supported |
| demo-001-c2 | 2 | 0.9702 | 0.0145 | 0.0153 | 0.0769 | `sufficient_consistent` | supported | supported | supported |
| demo-002-c1 | 1 | 0.0460 | 0.8280 | 0.1260 | 0.0000 | `sufficient_consistent` | refuted | refuted | refuted |
| demo-003-c1 | 1 | 0.0425 | 0.0425 | 0.9150 | 0.0000 | `insufficient` | undetermined | insufficient | insufficient |
| demo-004-c1 | 2 | 0.4081 | 0.5349 | 0.0570 | 0.6650 | `document_conflict` | refuted | conflicting | conflicting |

`gold_state` 只被原样带到结果里供事后对比，pipeline 的计算过程从不读取它。

### 输入数据的完整性要求

pipeline 在计算前做全量检查，任何一项不满足都直接报错，**不会**自动补全或
静默跳过：

| 异常 | 触发条件 |
| --- | --- |
| `NoClaimsError` | 样本没有 claim（本阶段不做自动 claim 抽取） |
| `MissingRelationPredictionError` | 某评估器未覆盖该 claim 的全部检索文档 |
| `DuplicateRelationPredictionError` | 同一 (sample, claim, doc, evaluator) 有多条预测 |
| `ReferentialIntegrityError` | 预测引用了不存在的 sample / claim / doc |
| `InconsistentEvaluatorReliabilityError` | 同一评估器在不同文档记录了不同可靠性 |

因此 `data/samples/mock_relations.jsonl` 必须覆盖 `demo.jsonl` 的**完整
claim × 文档网格**（当前 8 条）。缺失的预测不会被 `p_unknown=1` 补全，
也不会拿别的评估器结果顶替 —— 那会让「评估器没判」和「评估器判为不确定」
在结果里无法区分。

多评估器按**名称排序**依次融合，融合顺序只取决于评估器集合本身，与预测
文件的行序无关，结果可复现。

## 对照 baseline

代码在 `src/rag_ds/baselines/`，运行方式：

```bash
python scripts/run_baselines.py --config configs/baselines_demo.yaml --overwrite
```

### 四种方法

| 方法 | 规则 | 用到的可靠性 | 输出空间 |
| --- | --- | --- | --- |
| **Weighted Average** | 对全部评估器 × 全部文档的三个概率做加权平均 | 文档可靠性 × 评估器可靠性 | 三类 |
| **Majority Vote** | 每条关系预测一票，少数服从多数 | **都不用**（一票一权） | 三类 |
| **Single Evaluator** | 只用一个指定评估器，按文档可靠性加权平均 | 只用文档可靠性 | 三类 |
| **Conflict Aware** | 与 Weighted Average 完全相同的加权分数 + 显式冲突规则 | 文档可靠性 × 评估器可靠性 | **四类** |

Majority Vote 中单条预测内部出现并列最大值时投 `unknown` 票 —— 该条本身就
分不清方向，不替它选边。

样本**没有任何检索文档**时，四个 baseline 统一返回 `(0, 0, 1)` 与
`no_evidence`（与 D-S 侧的 `no_contexts` 处理对齐），Single Evaluator 也
不会因指定评估器没有预测而报错。反之，样本**有**检索文档但指定评估器
没有预测时，Single Evaluator 仍抛 `MissingBaselineEvaluatorError` —— 那是
数据完整性问题，不静默降级。

### 统一判定规则

```
1. 最高分 < decision_threshold      -> insufficient, below_threshold
2. 最高分之间差距 <= tie_tolerance  -> insufficient, score_tie
3. unknown 最高                      -> insufficient, unknown_highest
4. support 最高                      -> supported,    decided
5. refute 最高                       -> refuted,      decided
```

**阈值检查排在平局检查之前**：两条针锋相对的证据平均后常常同时「分数接近」
与「都不够高」，此时 `below_threshold`（整体信心不足）比 `score_tie` 更贴近
实际发生的事。

平局一律判 `insufficient`，不随机打破、不默认偏向 supported。

> 阈值 `decision_threshold = 0.5`、`tie_tolerance = 1e-6` **仅为调试值**。
> 正式实验必须在验证集上选择，测试集不得参与选择。

### 三个朴素 baseline 都不会输出 conflicting

这是**刻意的设计**，也是实验要展示的核心局限：朴素聚合把「两条针锋相对的
证据」压成一个低分或一个平局，无法与「谁都说不清楚」区分开。这三个方法的
`BaselinePrediction` 在模型层就禁止 `predicted_state = conflicting`。

D-S 方法用 `K_doc` 把这件事显式量化出来，因此能给出 `conflicting`。

### Conflict Aware：输出空间公平的第四个 baseline

前三个 baseline 无法输出 `conflicting`，意味着「D-S 四分类占优」总有一部分
功劳可以算在输出空间差异上。`conflict_aware` 在与 Weighted Average **完全相同
的加权分数**上加一条显式冲突规则，让 baseline 也拥有四类输出空间：

```
1. min(s, r) >= conflict_threshold 且 |s - r| <= conflict_margin
     -> conflicting, conflict_detected
2. 其余情况 -> 上面的标准级联
```

- 冲突检查**排在最前**：两派证据都足够强时，「它们互相矛盾」比「整体信心
  不足」更具体，与 D-S 混合区域映射为 CONFLICTING 的取舍一致；
- `conflict_margin` 要求两方向**势均力敌**（0.90 / 0.35 的一边倒不是冲突）；
- `conflict_threshold` 要求两方向**真的有分量**（0.05 / 0.05 的平局是无知）；
- 它不做任何 D-S 组合、不计算冲突量 K —— 看到的仍然只是三个平均后的分数。

`conflict_threshold` 与 `decision_threshold` 一样在验证集（或 CV 内折）上
做二维网格搜索；`conflict_margin` 固定为 0.1。**一个必须如实记录的现象**：
两套 CV 设定下搜出的最优 `conflict_threshold` 都顶在网格下界
（0.005–0.01）—— 这个 baseline 的最优策略是让冲突规则尽量放宽地触发，
它的分数因此依赖网格下界的具体取值，解读时应注意这一脆弱性。

### demo 上的实际对照

| claim | 标注 | D-S | Weighted Average | Majority Vote | Single Evaluator | Conflict Aware |
| --- | --- | --- | --- | --- | --- | --- |
| demo-001-c1 | supported | **supported** | insufficient | insufficient | insufficient | insufficient |
| demo-001-c2 | supported | **supported** | supported | supported | supported | supported |
| demo-002-c1 | refuted | **refuted** | refuted | refuted | refuted | refuted |
| demo-003-c1 | insufficient | **insufficient** | insufficient | insufficient | insufficient | insufficient |
| demo-004-c1 | conflicting | **conflicting** | insufficient | insufficient | insufficient | **conflicting** |

两处朴素 baseline 失手：

- **demo-004-c1**（标注 conflicting）—— 一条支持、一条反驳的文档被压成
  0.463 / 0.487 / 0.05，最高分不到 0.5，判为 `below_threshold`；投票则是
  1:1 的 `score_tie`。三个朴素 baseline 都只能说「不确定」，说不出「有冲突」。
  `conflict_aware` 的显式规则在这里能判对 —— 这正是它被引入的原因，
  也说明这一类样本上的差距确实来自输出空间，而不是融合机制。
- **demo-001-c1**（标注 supported）—— 一条强支持文档（0.90）与一条无信息
  文档（unknown 0.90）平均后得到 0.486 / 0.050 / 0.464，同样卡在阈值下方，
  `conflict_aware` 也一样失手（0.05 低于 conflict_threshold）。
  D-S 的 Dempster 组合让无信息证据自然让位，得到 m(S) = 0.856 ——
  **这一类样本上的差距才是融合机制本身的贡献**。

> 当前 baseline 使用**预设 mock 概率**，只验证代码流程是否正确，
> 不构成任何实验结论。

### 输出

| 文件 | 内容 |
| --- | --- |
| `outputs/predictions/demo_baselines.jsonl` | 每个 claim-method 一行 |
| `outputs/predictions/demo_baselines.csv` | 扁平摘要，UTF-8 with BOM |

输入完整性沿用 `rag_ds/integrity.py` 中 D-S pipeline 用的**同一份检查** ——
两条链路对「什么算合法输入」必须理解一致，否则实验对比就失去共同前提。

## 适配器层：接第三方评估器

`claim_extraction/` 与 `relation_evaluation/` 下的适配器**不 import
`ragchecker` / `ragas`，也不调用它们的任何 API，更不读取 API Key**。
这些库的函数签名随版本变化，凭记忆写出来的调用几乎一定是错的。
适配器只做一件事：把**你自己跑出来的结果**转换成本项目的统一格式。

| 模块 | 作用 |
| --- | --- |
| `claim_extraction/base.py` | `ClaimExtractor` 抽象接口 |
| `claim_extraction/mock.py` | 查表式假抽取器 |
| `claim_extraction/ragchecker_adapter.py` | RAGChecker claim 输出 → `Claim` |
| `relation_evaluation/ragchecker_adapter.py` | RAGChecker 关系标签 → `RelationPrediction` |
| `relation_evaluation/llm_evaluator.py` | 大模型评估器接口（调用逻辑由你实现） |
| `baselines/ragas_adapter.py` | RAGAS 分数读取，**保持原有粒度** |

接入流程：先按第三方库自己的文档跑出结果 → 写一小段胶水代码整理成本项目
定义的中间格式 → 交给适配器。RAGChecker 换版本时只有那段胶水要改。

### 标签到概率的映射

RAGChecker 给离散标签时，默认转换为：

```
entailment    -> (0.90, 0.05, 0.05)
contradiction -> (0.05, 0.90, 0.05)
neutral       -> (0.05, 0.05, 0.90)
```

> **这三组数字不是最终参数**，必须在验证集上校准。校准前得到的任何数字都
> 不能写进论文结论。若你的版本能给连续置信度，请直接用
> `prediction_from_probabilities` 传真实概率，不要先离散化。

### RAGAS 的粒度必须诚实

`Faithfulness` 是**答案级**指标。适配器用 `granularity` 字段显式记录粒度，
**绝不**把答案级分数复制到每条 claim 上冒充 claim-level 结果 —— 那会让
RAGAS 凭空获得「所有 claim 判断完全一致」的优势，是不公平比较。

## 接入 RAGChecker 真实输出

本节是把 RAGChecker 的真实判断接进 D-S 链路的完整流程。代码**不 import
`ragchecker`**，只读它写在磁盘上的结果文件。

### 依据的输出契约

以下三条取自 RAGChecker / RefChecker 源码，不是凭记忆写的：

| 事实 | 出处 |
| --- | --- |
| 输入为 `{"results": [{query_id, query, gt_answer, response, retrieved_context:[{doc_id, text}]}]}` | `examples/checking_inputs.json` |
| `retrieved2response: List[List[str]]`，形状 `[claim_num][doc_num]` | `ragchecker/container.py` 的 `RAGResult` |
| 标签只有 `Entailment` / `Neutral` / `Contradiction` | `refchecker/checker/checker_base.py` |
| `faithfulness` **只依赖** `retrieved2response` | `ragchecker/metrics.py` |

因此只需 `--metrics faithfulness` 就能拿到完整的 (claim, document) 网格，
`gt_answer` 不参与任何计算。导出脚本往 `gt_answer` 写的是一句显式占位说明，
**不是**把 claim 文本复制过去：万一有人误跑 `--metrics all_metrics`，得到的
precision / recall 会是一眼可见的垃圾值，而不是因为「gt_answer == response」
虚高到 1.0 的假好成绩。

### claim 对齐

RAGChecker 用自己的 extractor 从 `response` 抽 claim，抽出的条数 N 不一定
等于 1，而本项目的 `claim_id` 是数据集固定的：

| N | 处理 |
| --- | --- |
| 1 | 直接一一对应（CLIMATE-FEVER v1 的常见情形） |
| >1 | 把 N 个子 claim 的标签当作对同一条 claim 的 N 次投票，按标签分布对映射表做**凸组合**；N=1 时该式精确退化为查表 |
| 0 | **报错**。extractor 没抽出 claim 是失败，不会被伪造成 neutral |

`--strict-single-claim` 可要求 N 必须为 1，否则报错。

### 标签映射必须在验证集上校准

默认的 `(0.90, 0.05, 0.05)` 是**占位值**。`scripts/calibrate_label_mapping.py`
用 CLIMATE-FEVER 的人工证据投票作为验证集上的监督信号，对每个标签求条件平均：

```
mapping[L] = sum_pairs w_L(pair) * oracle_triple(pair) / sum_pairs w_L(pair)
```

即「RAGChecker 说 L 时，人工投票平均长什么样」。

> **论文必须写明**：映射表由**验证集**的人工投票标定，这是方法的一部分；
> **测试集全程不接触 oracle**，其关系概率完全由 RAGChecker 的标签换算而来。

某个标签在验证集上一次都没出现时直接报错，不静默沿用占位值 —— 否则「校准过
的参数」和「没校准的占位值」会在结果里混在一起看不出来。

### 两条判定路径

| 路径 | checker | 需要 LLM 凭证 | claim 来源 |
| --- | --- | --- | --- |
| **A. 本地 NLI**（当前采用） | RefChecker 自带 `NLIChecker` | 否，零费用离线 | 数据集给定，跳过 extractor |
| B. ragchecker-cli | 你指定的 LLM | 是 | LLM extractor 抽取 |

路径 A 用 `scripts/run_refchecker_nli.py`，跑在**独立的 `.venv-refchecker` 环境**里
—— refchecker 的依赖很重（torch / spacy / litellm / pytorch_lightning），装进主
venv 有可能顶掉主环境的 numpy 与 scikit-learn。

选 A 的理由不只是省钱：本数据集每个样本**恰好一条 claim**，且 `answer` 就是
claim 原文，extractor 在这里没有任何信息增益，却会带来「一句话被拆成 N 条」
的错位风险。跳过它之后 claim 对齐天然是 1:1。

判定环节与 RAGChecker 完全一致：同一个 `checker.check(...)`、同样
`merge_psg=False`、同样产出 `[claim_num][doc_num]` 的 `retrieved2response`。

> **论文里必须写成**「RefChecker NLI checker（`ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli`），
> claim 由数据集给定、未经 LLM 抽取」，**不能**写成「RAGChecker 全流程」。

### 五步流程

```powershell
# 0. 一次性：装好独立环境（CPU 版 torch，避免拉 2.5 GB 的 CUDA 包）
py -3.12 -m venv .venv-refchecker
.venv-refchecker\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cpu torch
.venv-refchecker\Scripts\python.exe -m pip install refchecker

# 1. 导出两个 split 的输入（不调模型）
python scripts/export_ragchecker_input.py --split validation
python scripts/export_ragchecker_input.py --split test

# 2. 跑本地 NLI checker，两个 split 各一次（首次会下载约 1.4 GB 模型）
.venv-refchecker\Scripts\python.exe scripts/run_refchecker_nli.py `
  --input outputs/ragchecker/validation_checking_inputs.json `
  --output outputs/ragchecker/validation_checking_outputs.json

# 3. 在验证集上校准标签映射
python scripts/calibrate_label_mapping.py `
  --outputs outputs/ragchecker/validation_checking_outputs.json

# 4. 摄取两个 split，产出关系文件与谱系清单
python scripts/ingest_ragchecker.py --split validation `
  --outputs outputs/ragchecker/validation_checking_outputs.json `
  --calibration outputs/metrics/ragchecker_label_calibration.json `
  --checker-name <模型>
python scripts/ingest_ragchecker.py --split test ...   # 同上，换 test

# 5a. 在验证集上重搜 D-S 阈值
python scripts/tune_thresholds.py --manifest data/processed/climate_fever_v1/manifest.json `
  --samples data/processed/climate_fever_v1/validation.jsonl `
  --predictions outputs/ragchecker/validation_relations_ragchecker.jsonl `
  --model-run outputs/ragchecker/validation_model_run.json --grid observed

# 5b. 给三个 baseline 同样的调参机会（不做这步，对比就不公平）
python scripts/tune_baseline_thresholds.py `
  --predictions outputs/ragchecker/validation_relations_refchecker_nli.jsonl `
  --model-run outputs/ragchecker/validation_model_run.json `
  --single-evaluator refchecker_nli

# 6. 把两组阈值手工填回配置，跑测试集，出图
python scripts/run_experiment.py --config configs/climate_fever_refchecker_nli_test.yaml
python scripts/export_results.py --config configs/climate_fever_refchecker_nli_test.yaml `
  --threshold-search outputs/metrics/refchecker_nli_threshold_search.json
```

### baseline 必须和 D-S 用同一套调参协议

D-S 的门控阈值在验证集上搜过；baseline 若还用着 `decision_threshold = 0.5`
这个调试默认值，两边就不在同一条件下比较 —— **那不是 baseline 弱，是 baseline
没调参**。实测差别是决定性的：校准后的概率偏软，最高分几乎都够不到 0.5，
`weighted_average` 与 `single_evaluator` 会把测试集 80 条**全部**判成
`insufficient`，Macro-F1 退化成 0.1000。

`scripts/tune_baseline_thresholds.py` 用与 D-S 相同的协议（同一验证集、同样以
Macro-F1 为目标）为每个方法搜索阈值，并给出三个方法最优**平台的交集** ——
交集非空时填一个值就能让三者同时处于各自最优。Macro-F1 常在整段阈值上持平，
取平台**中位数**而不是边缘值（边缘再动一点就掉出平台）。

先用 `--limit 5` 跑通一遍再跑完整 split：导出脚本和 `run_refchecker_nli.py`
都有 `--limit`，摄取脚本的 `--allow-partial` 专为这种冒烟测试准备，它会强制
进入 dry-run，不写任何文件。

### 关系文件的谱系：oracle 与模型输入分家

数据集清单里登记的 `*_relations.jsonl` 是人工 oracle，`verify_split_artifacts`
要求关系文件**就是**登记的那一份。模型产出的关系不属于数据集，因此另配一份
`ModelRunManifest`（`src/rag_ds/model_runs.py`），它同时钉住：

- 这批关系属于哪个 split 的哪份样本（samples 摘要须与数据集清单一致）；
- 关系文件本身的摘要与记录数；
- 产出它的 evaluator、extractor / checker 模型名、原始输出文件摘要；
- 用的标签映射表及其来源（`calibrated_on_validation` / `placeholder_default`）。

`relation_predictions_kind` 固定为 `model_prediction`，与数据集清单的
`annotation_oracle` 形成对照。`run_experiment.py` 与 `tune_thresholds.py` 读到
模型清单时会打印模型名；映射表没校准时会明确警告**结果不能写进论文**。

### 换了关系输入，阈值必须重搜

`configs/climate_fever_oracle_test.yaml` 里的 `theta=0.3`、`K_doc=0.6` 是在
**oracle 关系**上搜出来的，不能直接套用到 RAGChecker 的关系上。

更要紧的是**网格范围本身**：固定网格 `theta ∈ (0.3, ..., 0.7)` 隐含「融合后的
m_theta 会落在 0.3 以上」这个假设。校准后的概率比占位值软得多，而 Dempster
组合在 5 篇文档上会把 m(Θ) 连乘压下去 —— 用合成数据彩排时，m_theta 的实际
范围是 `[0.002, 0.143]`，**整条 theta 轴全部落在观测范围之外，门控一次都没有
触发**，而搜索结果看上去毫无异常（所有网格点 Macro-F1 完全相同，消融里
`no_theta_gate` 的 Δ 恰好为 0）。

因此 `tune_thresholds.py` 增加了两样东西：

- `--grid observed`：按验证集上**实际观测到的** m_theta / K_doc 分位数构造网格；
- 网格健康检查：候选阈值整体落在观测范围之外、最优值落在网格边界、或所有网格点
  得分完全相同时，都会明确报警。

默认仍是 `fixed` 网格，既有 oracle 结果保持可复现。

### 第一次真实结果（CLIMATE-FEVER test，80 条 claim）

关系输入 = RefChecker NLI checker，标签映射在验证集上校准，D-S 与三个 baseline
的阈值都在验证集上按同一协议搜出。谱系见 `outputs/ragchecker/test_model_run.json`。

**实验 1 —— 四分类**

| 方法 | Accuracy | Macro-F1 |
| --- | --- | --- |
| **D-S** | 0.3500 | **0.3212** |
| Weighted Average | 0.3500 | 0.2640 |
| Majority Vote | 0.3500 | 0.2486 |
| Single Evaluator | 0.3500 | 0.2640 |

**实验 2 —— 证据不足识别**：`m_theta` AUROC = 0.5583，三个 baseline 分别是
0.5558 / 0.5608 / 0.5558。**全部贴近随机，D-S 在这条轴上没有优势。**

**实验 3 —— 文档冲突识别**：`K_doc` AUROC = **0.7308**，baseline 最好的是
0.6871（Majority Vote 只有 0.4138，低于随机）。**这是本方法唯一站得住的优势。**

> **已被推翻**：这里 baseline 的冲突分数是 `1 - |s - r|`，所有文档都说不清时
> 它也接近 1，等于把「证据不足」当成冲突，是过弱的对手。换成定义合理的简单
> 统计量后 `K_doc` 与它们打平，见「五、m_theta / K_doc 作为检测分数不优于
> 简单统计量」。

**实验 4 —— 消融**

| 变体 | Macro-F1 | Δ |
| --- | --- | --- |
| full | 0.3212 | — |
| no_reliability | 0.3212 | +0.0000（恒等变换，Δ 无意义） |
| no_theta_gate | 0.3525 | **+0.0313** |
| no_doc_conflict_gate | 0.2635 | −0.0576 |
| no_two_dimensional_gate | 0.2527 | −0.0684 |

去掉 theta 门控反而**更好**，去掉 K_doc 门控明显变差 —— 与实验 2、3 的结论一致：
冲突这条轴有效，证据不足那条轴在当前关系输入下无效。

### 换 NLI 模型时的一个静默陷阱

RefChecker 的 `nli_checker.py` 里写着：

```python
LABELS = ["Entailment", "Neutral", "Contradiction"]
...
ret = [LABELS[p] for p in batch_preds]      # p 是 argmax 下标
```

它**直接用硬编码的顺序去索引**，从没查过模型自己的 `config.id2label`。而各家
NLI 模型的类别顺序并不一致，实测四个常用模型出现了**三种**顺序：

| 模型 | `id2label` 顺序 |
| --- | --- |
| `ynie/roberta-large-...-nli`（RefChecker 默认） | `[entailment, neutral, contradiction]` |
| `microsoft/deberta-large-mnli` | `[CONTRADICTION, NEUTRAL, ENTAILMENT]` |
| `FacebookAI/roberta-large-mnli` | `[CONTRADICTION, NEUTRAL, ENTAILMENT]` |
| `cross-encoder/nli-deberta-v3-base` | `[contradiction, entailment, neutral]` |

也就是说，把默认模型以外的模型交给 RefChecker 的 `NLIChecker`，**entailment 与
contradiction 会被静默对调** —— 不抛异常、不打警告、数字照常产出。用 DeBERTa
跑出来的"支持"其实全是"反驳"。

`scripts/run_refchecker_nli.py` 因此一律以模型 `config.id2label` 为准重新取标签：

- 顺序与 RefChecker 的假设一致时，额外拿 `check()` 的返回值**逐条交叉验证**；
- 不一致时打出醒目警告，按模型真实顺序重算，并把
  `refchecker_order_matches: false` 写进输出的 `rag_ds_run_info`。

输出里的概率三元组一律按 `(Entailment, Neutral, Contradiction)` 排列，与模型
内部顺序无关，下游无需关心这件事。

### 离散标签 vs 连续概率：两个变体的对照

**离散路径的结构性缺陷。** checker 每篇文档只输出 3 个标签之一、共 5 篇文档；
Dempster 组合与 `K_doc = 1 - ∏(1-K_i)` 都与顺序无关，于是诊断结果**只取决于
(n_E, n_N, n_C) 三个计数**，上限只有 C(7,2) = 21 个格点。实测 80 条 claim 落在
**11 个点**上，其中 42 条（52.5%）全部落在同一个点（5 篇文档全 Neutral）——
这 42 条里四类金标准都有，**任何阈值都分不开它们**，构成不可约的误差下界。

NLI 模型本身输出的就是三类 softmax，正好对应 `(p_support, p_unknown, p_refute)`，
只是 RefChecker 在 `nli_checker.py` 里用 `argmax` 丢掉了。
`run_refchecker_nli.py --emit-probabilities` 把它捞回来（复用同一份已加载的模型，
并**逐条断言 argmax 与 `check()` 返回的标签一致**），
`ingest_ragchecker.py --use-probabilities` 直接用连续值，完全绕开映射表。

**两个变体，同一套协议**（阈值与 baseline 阈值都在验证集上搜）：

| 指标（test，80 条） | A 离散标签 + 校准映射 | B 连续 softmax |
| --- | --- | --- |
| 不同的诊断点 | 11 | **80** |
| D-S Accuracy | 0.3500 | **0.3625** |
| D-S Macro-F1 | 0.3212 | **0.3333** |
| 最好的 baseline Macro-F1 | 0.2640 | 0.2519 |
| `m_theta` AUROC（证据不足） | 0.5583 | **0.5975** |
| `K_doc` AUROC（文档冲突） | **0.7308** | 0.6900 |
| `supported` 答对数 | **0 / 20** | 3 / 20 |

**消融结果发生了质变**，这是两者最重要的差别：

| 变体 | A 离散 Δ | B 连续 Δ |
| --- | --- | --- |
| no_theta_gate | **+0.0313**（门控有害） | **−0.0687**（门控有效） |
| no_doc_conflict_gate | −0.0576 | −0.0601 |
| no_two_dimensional_gate | −0.0684 | **−0.1316** |

离散路径下 theta 门控是**负作用**（去掉反而更好）；连续路径下两个门控都有正贡献，
同时去掉损失最大。**支撑「二维门控各自有用」这一论点的是连续概率版本。**
v2 的 616 条嵌套交叉验证（每个变体各自重搜阈值）已确认两个维度各自显著有用，
见「六、v2 嵌套消融」。

代价是 `K_doc` 的冲突识别 AUROC 从 0.7308 降到 0.6900。论文里两组数字都应报告。

### v2 与嵌套交叉验证：主结论的统计证据

v1 测试集只有 80 条 claim，D-S 相对最好 baseline 的 Macro-F1 差异（+0.081）
95% 置信区间跨 0（p = 0.166），**单次留出划分无法支撑「显著优于」**。
v2（`data/processed/climate_fever_v2/`，616 条、四类各 154）配合嵌套交叉验证
解决了这个问题：5 个外折，阈值只在每个外折的其余各折上搜，测试折全程不参与；
每条 claim 恰好被预测一次，n = 616，区间随 √n 收窄。

v2 的 RefChecker NLI 推理用 `scripts/run_refchecker_nli_chunked.py` 完成
（CPU 环境下分轮断点续跑，输出与 `run_refchecker_nli.py` 同形；三个 split
的标签分布均为 Neutral 约 77%、Contradiction 约 19%、Entailment 约 3.5%），
摄取时用 `--use-probabilities` 连续概率路径。运行方式：

```powershell
python scripts/run_cross_validation.py `
  --manifest data/processed/climate_fever_v2/manifest.json `
  --relations-template "outputs/ragchecker_v2/{split}_relations_{evaluator}_probs.jsonl" `
  --model-run-template "outputs/ragchecker_v2/{split}_model_run_{evaluator}_probs.json" `
  --evaluator refchecker_nli --single-evaluator refchecker_nli
```

**结果（616 条 claim，阈值逐折重选）：**

| 方法 | Accuracy | Macro-F1 | 95% CI |
| --- | --- | --- | --- |
| **D-S** | 0.3750 | **0.3535** | [0.3145, 0.3932] |
| Conflict Aware | 0.3653 | 0.3074 | [0.2738, 0.3419] |
| Weighted Average | 0.3442 | 0.2512 | [0.2242, 0.2773] |
| Single Evaluator | 0.3442 | 0.2512 | [0.2242, 0.2773] |
| Majority Vote | 0.3425 | 0.2496 | [0.2229, 0.2759] |

与 D-S 的配对自助比较（5000 次重采样）：

| 对手 | Δ Macro-F1 | 95% CI | p |
| --- | --- | --- | --- |
| Weighted Average | +0.1023 | [+0.0587, +0.1442] | < 0.001 |
| Single Evaluator | +0.1023 | [+0.0587, +0.1442] | < 0.001 |
| Majority Vote | +0.1039 | [+0.0600, +0.1462] | < 0.001 |
| Conflict Aware | +0.0461 | [+0.0007, +0.0906] | 0.048 |

**所有区间都不跨 0，主结论「D-S 显著优于聚合类 baseline」成立。**
各折选出的 theta 在 [0.69, 0.88]、K_doc 在 [0.03, 0.14] 之间小幅波动，
baseline 判定阈值五折一致（0.475），结论对折划分不敏感。逐折明细与逐条
预测见 `outputs/metrics/cross_validation/`。

**优势的分解（本节最重要的读法）：**

- D-S 对朴素 baseline 的总优势 ≈ **+0.102**；
- 其中 **输出空间**（能输出 conflicting）贡献 ≈ +0.056
  （conflict_aware 0.3074 − weighted_average 0.2512）；
- 剩下的 **融合机制**贡献 ≈ +0.046（0.3535 − 0.3074），仍然显著
  （p = 0.048），但已不到总优势的一半。

论文里必须同时报告这三层数字：只拿 D-S 对朴素 baseline 的 +0.102 说事，
会把输出空间差异误记成机制优势。

> **更正**：上面的 conflict_aware 只单独调了冲突下限（margin 固定 0.1，判定
> 阈值与三个朴素 baseline 共用）。三个参数都单独调好后它升到 0.3374，D-S 对它的
> Δ 降到 +0.016（p = 0.41）；判定规则相同、不用 D-S 的 Statistic Gate 与 D-S
> 打平。「+0.046 来自融合机制」这一分解**不再成立**，见「四」。

注意 v2 的单评估器设定下 K_eval 恒为 0、weighted_average 与
single_evaluator 数值相同，且三个朴素 baseline 结构上仍不能输出
conflicting（conflict_aware 可以）—— 其余限制见下节。

v1（80 条）两个变体的显著性检验在
`outputs/metrics/refchecker_nli/significance.json`（离散）与
`outputs/metrics/refchecker_nli_probs/significance.json`（连续），均由
`scripts/run_significance.py` 产出。离散版对 Majority Vote 显著
（p = 0.034）、对 Weighted Average / Single Evaluator 不显著（p = 0.188）；
连续版对三个 baseline 均不显著（最好 p = 0.148）—— v1 的数字只能作为
初步结果引用，正式结论以上面的嵌套交叉验证为准。

### 多评估器：K_eval 第一次有真实取值

第二个评估器为 `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`（MNLI+FEVER+ANLI
训练，与 ynie roberta-large 同源但不同架构；base 规模，CPU 上约快 2.5 倍），
同样走 RefChecker NLIChecker + 连续概率路径，摄取为 `refchecker_nli_deberta`。
其 `id2label` 顺序恰好与 RefChecker 硬编码一致；若换其他模型，脚本会以模型
`config.id2label` 为准重取标签并交叉验证。

双评估器嵌套交叉验证（协议与单评估器完全相同，阈值逐折重选）：

```powershell
python scripts/run_cross_validation.py `
  --manifest data/processed/climate_fever_v2/manifest.json `
  --relations-template "outputs/ragchecker_v2/{split}_relations_{evaluator}_probs.jsonl" `
  --model-run-template "outputs/ragchecker_v2/{split}_model_run_{evaluator}_probs.json" `
  --evaluator refchecker_nli refchecker_nli_deberta `
  --single-evaluator refchecker_nli `
  --out-dir outputs/metrics/cross_validation_2eval
```

| 方法 | Accuracy | Macro-F1 | 95% CI |
| --- | --- | --- | --- |
| **D-S** | 0.3523 | **0.3320** | [0.2934, 0.3708] |
| Conflict Aware | 0.3312 | 0.2856 | [0.2535, 0.3167] |
| Weighted Average | 0.3474 | 0.2531 | [0.2265, 0.2793] |
| Single Evaluator | 0.3442 | 0.2510 | [0.2239, 0.2772] |
| Majority Vote | 0.3377 | 0.2452 | [0.2182, 0.2720] |

D-S 对三个朴素 baseline 的配对比较均为 **p < 0.001**（Δ 在 +0.079 到
+0.087 之间），对 conflict_aware 为 **Δ = +0.0464，p = 0.040**
（CI [+0.003, +0.092]，不跨 0）。分解与单评估器一致：总优势里约一半来自
输出空间，一半来自融合机制，两层各自都仍显著。（这里的 conflict_aware 同样
没有调充分；调充分后，单评估器、双评估器、三评估器三种设定下 D-S 对它的差异
都不再显著，见「四」。）三点必须如实报告：

1. **加入第二个评估器后 D-S 的 Macro-F1 从 0.3535 降到 0.3320，但这个下降
   不显著**（配对 Δ = −0.0215，95% CI [−0.055, +0.011]，p = 0.196）。能说的
   只是「多评估器融合没有带来增益」，不能说「显著变差」。完整的多评估器实验
   与原因分析见下文「二、多评估器融合为什么没有增益」。
2. **K_eval 机制已真实激活，但告警尚无判别力**：616 条 claim 中 23 条
   （3.7%）触发 `evaluator_disagreement`（默认阈值 0.4），K_eval 最大 0.92；
   但触发组的错误率（60.9%）与未触发组（63.1%）没有差别。K_eval 作为
   「这条结论可疑」的信号在本数据上还没有实用价值，论文里不能把它写成
   已验证的优势。
3. **conflict_aware 的最优 conflict_threshold 顶在网格下界**（各折
   0.005–0.01）：这个 baseline 的最优策略是让冲突规则尽量放宽地触发，
   其分数对网格下界敏感；且 D-S 对它的优势（p ≈ 0.04–0.05）远小于对朴素
   baseline（p < 0.001），机制优势是「显著但不大」，不能写成「大幅领先」。

双评估器下 weighted_average 与 single_evaluator 不再等价（前者融合两个
评估器，后者只用 roberta），加上 conflict_aware，对照方法有了四个真实
变体。

> **注意一处脆弱性**：连续版 baseline 搜索中，`weighted_average` 与
> `single_evaluator` 的最优平台只有 `0.54` 这一个网格点（`majority_vote` 的平台是
> `[0.02, 0.6]`）。单点最优容易在测试集上掉下来，需如实说明。

### 仍然存在的两个限制

1. **`supported` 类几乎无法识别**：NLI checker 在测试集 400 个组合里只给出 9 次
   Entailment（2.2%），支持质量累积不起来。连续概率把答对数从 0 提到 3，
   但仍远低于其他三类。瓶颈在关系评估器本身，不在 D-S 融合。
2. **`no_reliability` 消融无法进行**：数据集里 2000 个 `reliability` 全是 1.0，
   `retrieval_score` 全是 `None`，去折扣是恒等变换。结果里的 `is_vacuous`
   字段会标出来。这条改用下面的敏感性分析来补。

### 可靠性折扣的敏感性分析

既然 `no_reliability` 消融在本数据集上是空转，可靠性折扣这个核心机制就不能留成
空白。`scripts/run_reliability_sensitivity.py` 用两组设定把它实际跑起来：

```powershell
python scripts/run_reliability_sensitivity.py `
  --config configs/climate_fever_refchecker_nli_probs_test.yaml `
  --provenance data/processed/climate_fever_v1/test_provenance.jsonl `
  --figure outputs/figures/refchecker_nli_probs/reliability_sensitivity.png
```

**S1 —— 均匀可靠性扫描（无 oracle，可直接报告）**

把所有文档可靠性统一设为 r 并扫 r。折扣只把质量从确定焦元移向 Theta，因此理论上
`m_theta` 必须单调上升、`K_doc` 必须单调下降。连续概率版实测（test，80 条）：

| r | Macro-F1 | Δ | m_theta 均值 | K_doc 均值 |
| --- | --- | --- | --- | --- |
| 1.00 | 0.3333 | — | 0.4424 | 0.0495 |
| 0.80 | **0.3430** | **+0.0098** | 0.4954 | 0.0357 |
| 0.60 | 0.3147 | −0.0185 | 0.5642 | 0.0227 |
| 0.40 | 0.2715 | −0.0618 | 0.6583 | 0.0114 |
| 0.20 | 0.2582 | −0.0751 | 0.7944 | 0.0032 |
| 0.10 | 0.2616 | −0.0716 | 0.8863 | 0.0008 |

两条单调性在全部 10 个取值上都成立 —— 这是折扣实现**在整条链路上**（而不只是
单元测试的小例子上）行为正确的证据。Macro-F1 在 r≈0.8 处有一个很小的峰值
（+0.0098），之后随 r 下降持续退化。

**S2 —— 标注一致度作为可靠性（oracle 上界，不是模型性能）**

CLIMATE-FEVER 为每段证据记录了标注投票与熵值，据此取 `r = 1 - H / ln(3)`：
一致度高的证据给高可靠性。实测该信号有真实差异（7 种取值，400 篇中 208 篇为
1.0，最低 0.0，均值 0.70），但结果是：

| 设定 | Macro-F1 | Δ |
| --- | --- | --- |
| oracle_vote_agreement | 0.3349 | **+0.0016** |

**即便给一个来自人工标注的完美可靠性信号，也只改变 80 条里的 2 条预测、
Macro-F1 只涨 0.0016。** 这比「消融是空转」强得多：它给出的结论是
**在 CLIMATE-FEVER 上，文档级可靠性差异不是限制因素**，而不是「没测」。

> 离散标签版的同一分析还暴露了额外的脆弱性：r ≤ 0.8 时 Macro-F1 直接塌到
> 0.1000（退化成全判一类）—— 它的阈值只在 r=1.0 那个窄带里有效。
> 连续概率版在整个扫描区间里都保持在 0.26 以上。

两组都是敏感性分析，不是主结果；CSV 里的 `is_oracle` 列把 S2 标了出来。

## 指标与实验

### 标签口径（必须写进论文）

金标准有四类，但两侧方法的输出空间不同：D-S 可以给出全部四类（外加
`undetermined`），三个 baseline **在结构上无法输出 `conflicting`**。
本项目的处理是：

1. **混淆矩阵**用完整标签集（四类 + `undetermined`），方阵，不隐藏任何一格；
2. **Macro-F1 默认只在「金标准中实际出现过的类」上平均** —— baseline 在
   `conflicting` 上的 0 分**照常计入**，那是它真实的能力缺口。

需要「只比三类」的补充视角时，显式传 `macro_labels`，并同时给出两套数字。

### 三个脚本

```bash
python scripts/run_experiment.py --config configs/experiment.yaml --overwrite
```

```bash
python scripts/tune_thresholds.py --manifest data/processed/climate_fever_v1/manifest.json --samples data/processed/climate_fever_v1/validation.jsonl --predictions data/processed/climate_fever_v1/validation_relations.jsonl
```

```bash
python scripts/export_results.py --config configs/experiment.yaml
```

产出：

| 文件 | 内容 |
| --- | --- |
| `outputs/metrics/main_results.csv` | 实验 1–3（四分类 / 证据不足识别 / 冲突识别） |
| `outputs/metrics/ablation_results.csv` | 实验 4 消融 |
| `outputs/metrics/threshold_search.json` | 阈值搜索全部网格点 |
| `outputs/figures/confusion_matrix.png` | D-S 混淆矩阵 |
| `outputs/figures/diagnostic_scatter.png` | 二维诊断散点图（x=m_theta, y=K_doc） |
| `outputs/figures/threshold_sensitivity.png` | 阈值敏感性曲线 |

图内文字一律英文：matplotlib 自带字体没有中文字形，用中文会渲染成方框，
而依赖系统中文字体又会让图在别的机器上画不出来。

### 阈值只能在验证集上搜

`search_thresholds` 的签名强制传入 `SplitName`，传 `TEST` 或 `TRAIN` 会
**直接报错**。搜索完成后需要**手工**把最优阈值填回 `configs/experiment.yaml`
并锁定 —— 刻意不自动改写配置，避免「什么时候用了哪组阈值」变成糊涂账。

搜索只重跑最后一步门控（阈值不影响前面的 BPA、折扣与融合），因此 25 个
四分类网格点只需把 D-S 链路跑一次。`K_eval` 只翻转 `evaluator_disagreement`
告警，不改变 `primary_state`，所以**不进入** Macro-F1 网格，也不进入默认的
分类消融（`no_eval_conflict_alert` 会被直接拒绝）。绘图脚本读取已保存的
验证集搜索结果，不会用测试数据重新搜索。

### 依赖分层

`import rag_ds` 只拉起 pydantic 与 PyYAML（约 0.25 秒）。
`metrics` / `tuning` / `experiments` 会引入 scikit-learn 与 matplotlib，
需要显式子包导入：

```python
from rag_ds.experiments import run_comparison
from rag_ds.metrics import classification_report
from rag_ds.tuning import search_thresholds
```

## 必须正视的问题及其处理

### 一、主结论对分折种子的敏感性

单次 5 折 CV 给出 D-S 相对最强 baseline（`conflict_aware`）的 Δ Macro-F1 =
+0.0461，95% 区间 `[+0.0007, +0.0906]` —— 下界离 0 只有 0.0007。这种"刚好
显著"必须做稳健性检验（`scripts/run_cv_seed_robustness.py`，5 个分折种子，
**全部报告，不挑**）：

| 对手 | Δ 均值 | Δ 标准差 | Δ 范围 | 显著种子数 |
| --- | --- | --- | --- | --- |
| `conflict_aware` | **+0.0522** | 0.0067 | [+0.0461, +0.0637] | **5/5** |
| `majority_vote` | +0.1090 | 0.0063 | [+0.1039, +0.1212] | 5/5 |
| `single_evaluator` | +0.1075 | 0.0066 | [+0.1023, +0.1203] | 5/5 |
| `weighted_average` | +0.1075 | 0.0066 | [+0.1023, +0.1203] | 5/5 |

D-S 的 Macro-F1 均值 0.3586、标准差 0.0063。**种子 42 的 +0.0461 是五个里最小
的那个**，结论不随划分翻转。

> **措辞限制**：五次 CV 共用同一批 616 条样本、只是换了划分，因此这衡量的是
> **划分敏感性**，不是独立重复实验。论文里不能写成"重复五次均显著"。

> **后续更正**：这张表里的 `conflict_aware` 是只调了冲突下限的版本。三个参数都
> 调好之后，同样 5 个种子上 D-S 对它的优势**一个都不显著**（Δ 均值 +0.0198，
> 见「四」），因此「对 conflict_aware 5/5 显著」不能再作为主结论引用。

### 二、多评估器融合为什么没有增益

**先纠正一个早期结论。** 单评估器 0.3535 → 双评估器 0.3320 这个下降曾被当成
需要解释的真实效应，并用「两个 NLI 模型误差高度相关（Cohen's κ = 0.62），违反
Dempster 规则的独立性假设」来解释。两件事推翻了这个说法：

1. 这个下降**本身就不显著**（配对 Δ = −0.0215，p = 0.196）；
2. 引入与两个 NLI 模型明显更不相关的第三个评估器（flan-t5，κ ≈ 0.44）后，
   等权融合**掉得更多**，而不是更少。若独立性是主因，结果应当相反。

**完整实验**（v2，616 条，5 折嵌套 CV，同一分折种子，所以可配对比较）：

| 评估器组合 | 评估器间 κ | 等权融合 | 内层估计可靠性 |
| --- | --- | --- | --- |
| roberta 单独 | — | **0.3535** | 0.3535 |
| deberta 单独 | — | 0.3398 | — |
| flan-t5 单独 | — | 0.3245 | — |
| roberta + deberta | 0.61 | 0.3320 | 0.3466 |
| roberta + flan-t5 | 0.45 | 0.3122 | 0.3452 |
| 三者 | — | 0.3302 | 0.3467 |

配对自助检验（D-S Macro-F1，5000 次）：

| 比较 | Δ | 95% CI | p |
| --- | --- | --- | --- |
| roberta+deberta 等权 vs roberta 单独 | −0.0215 | [−0.055, +0.011] | 0.196 |
| **roberta+flan-t5 等权 vs roberta 单独** | **−0.0414** | **[−0.078, −0.003]** | **0.032** |
| 三者等权 vs roberta 单独 | −0.0233 | [−0.059, +0.012] | 0.193 |
| roberta+flan-t5：估计可靠性 vs 等权 | +0.0330 | [+0.0001, +0.066] | 0.050 |
| roberta+deberta：估计可靠性 vs 等权 | +0.0146 | [−0.011, +0.040] | 0.273 |
| 三者：估计可靠性 vs 等权 | +0.0165 | [−0.001, +0.036] | 0.070 |
| 三组「估计可靠性」vs roberta 单独 | ≈ −0.007 | 均跨 0 | 0.60–0.69 |

**统计上站得住的结论只有三条：**

1. **唯一显著的退化**来自等权融合一个明显更弱的评估器（flan-t5 单独只有
   0.3245）。全部评估器的 `evaluator_reliability` 都是 1.0，弱评估器拿到了和
   强评估器同样的话语权 —— 这是「质量不等却等权」，不是「证据不独立」。
2. **内层估计可靠性消除了这个退化**：估计后 roberta+flan-t5 与 roberta 单独
   已无显著差异。这是**可靠性折扣机制在真实数据上第一次起作用的证据**
   （此前它在 CLIMATE-FEVER 上一直是空转，见「可靠性折扣的敏感性分析」）。
   但「估计 vs 等权」的改进 CI 下界只有 +0.0001，是**边缘显著**，措辞要克制。
   而且这不是 D-S 独有的：把同一份估计出来的可靠性交给不用 D-S 的 Statistic
   Gate 作权重，它与 D-S 打平（12 组对比全部不显著，见「四」）。
3. **没有任何多评估器组合优于最好的单评估器。**

估计器给三个评估器的可靠性分别约为 0.44 / 0.42 / 0.39（逐折），彼此差距远小于
它们的实际质量差距 —— 基于 AUROC 的估计器压缩了质量差异，所以弱评估器仍分到
相当可观的权重，只能挽回一部分损失。

独立性问题并非不存在（两个 NLI 模型 κ = 0.62，三个概率通道 r = 0.71–0.74），
但本实验**无法把它与质量差异分离开**，论文里不能把它写成已证实的原因。

### flan-t5 评判器：一个已修正的实现缺陷

第三个评估器的早期版本（`scripts/run_t5_judge_chunked.py`）有两个缺陷，
导致 validation 的 620 条里 99.8% 判为 unknown、0% 判为 support，与人工
oracle 的 Cohen's κ = +0.002（随机水平）：

1. **长度偏置**：对候选词 `entailment` / `neutral` / `contradiction` 的各
   token 对数概率**求和**。`entailment` 在 flan-t5 分词器下是 4 个 token，
   另两个各 1 个，求和使它在结构上几乎不可能胜出；
2. **提示格式**：`nli premise: ... hypothesis: ...` 不是 FLAN 的原生模板。

只把求和改成求平均**修不好** —— 在 200 条子集上它会从「全判 neutral」翻成
「全判 support」，κ 仍为 0。修正版改用 FLAN 训练时用过的 MNLI 指令模板，
取三个选项（`yes` / `it is not possible to tell` / `no`）首 token 的 logit
打分，每个首词都是单个 token，天然没有长度偏置，且每个组合只需一次前向：

| 评估器（v2 validation，620 条） | κ(oracle) | r(support) | r(refute) | S / R / U 分布 |
| --- | --- | --- | --- | --- |
| roberta | +0.130 | +0.214 | +0.362 | 3.7 / 18.2 / 78.1% |
| deberta | +0.111 | +0.188 | +0.330 | 4.4 / 25.0 / 70.6% |
| flan-t5（修正后） | +0.071 | **+0.238** | +0.236 | 3.9 / 28.7 / 67.4% |
| flan-t5（作废版） | +0.002 | +0.006 | +0.048 | 0.0 / 0.2 / 99.8% |

作废输出移到了 `outputs/ragchecker_v2/_invalid_flan_t5_sum_logprob/`（附说明）；
修正版评估器名改为 `flan_t5_mnli`，打分版本写进每条进度记录，续跑时版本不一致
会直接拒绝，防止新旧两种打分混进同一个文件。它的关系文件不经过 `ingest`，
谱系清单由 `scripts/register_model_run.py` 登记（登记前校验网格完整、
单一评估器、与 run_info 一致）。

### 三、选择性回答分析里的两个 bug

**Bug A：指标方向写反了。** 早期脚本计算的是
`trapezoid(accuracy, coverage)` —— **准确率曲线下面积，越高越好** —— 却把列名
写成 `aurc` 并注明"越低越好"。按错误方向读会把 D-S 的最好成绩读成最差。

**Bug B：风险定义对 D-S 系统性不公。** 原定义
`risk = max(insufficiency_score, conflict_score)` 中，`m_theta` 与 `K_doc`
并不是不确定性，而是四类里**两类的证据本身**：`m_theta` 高意味着 D-S 有把握判
`insufficient`，`K_doc` 高意味着有把握判 `conflicting`。实测 616 条里 170 条
`insufficient` 预测的风险分均值高达 0.94，被整体顶到弃答队列最前 —— 排序实际
在排"是不是判了 insufficient"。

修正：新增按预测类别取该类证据的置信度（见
`src/rag_ds/experiments/selective_confidence.py`），两侧对称；**两套定义并列
报告**，因为它们问的是不同的问题。

| 风险定义 | 方法 | AURC ↓ | AUROC ↑ |
| --- | --- | --- | --- |
| `max_signal` | **ds** | **0.6115** | 0.5761 |
| `max_signal` | `weighted_average` | 0.6438 | **0.5930** |
| `class_conditional` | **ds** | **0.5924** | **0.6212** |
| `class_conditional` | `majority_vote` | 0.6402 | 0.5363 |
| `class_conditional` | `weighted_average` | 0.6468 | 0.5235 |

**点估计上** D-S 在 `class_conditional` 下两个指标都最好，但只看点估计会说过头。
配对自助检验（`selective_significance.csv`，2000 次，claim 级配对重采样；
ΔAUROC > 0、ΔAURC < 0 表示 D-S 占优，`*` = 95% 区间不跨 0）：

| 风险定义 / 设定 | 对手 | ΔAUROC | ΔAURC |
| --- | --- | --- | --- |
| `class_conditional` / 单评估器 | `conflict_aware` | **+0.108\*** | −0.036 |
| | `weighted_average` | **+0.098\*** | **−0.054\*** |
| | `majority_vote` | **+0.085\*** | **−0.048\*** |
| `class_conditional` / 双评估器 | `conflict_aware` | +0.052 | −0.025 |
| | `weighted_average` | **+0.093\*** | −0.023 |
| | `majority_vote` | **+0.077\*** | −0.022 |
| `max_signal` / 单评估器 | 全部四个 | 均不显著 | 均不显著 |
| `max_signal` / 双评估器 | `conflict_aware` | **−0.065\***（D-S 更差） | −0.005 |
| | `weighted_average` | **−0.041\***（D-S 更差） | +0.004 |

能写进论文的准确表述：

- 按标准口径（`class_conditional`），D-S 置信度对"判错"的**排序能力**（AUROC）
  显著优于三个朴素 baseline，单评估器下也显著优于 `conflict_aware`；
- 弃答后的**准确率收益**（AURC）只在单评估器下、且只对三个朴素 baseline 显著，
  对 `conflict_aware` 不显著；双评估器下对谁都不显著；
- 按 `max_signal` 口径，D-S 没有优势，双评估器下 AUROC 甚至**显著更差**。
  这一行不能省略 —— 它说明结论依赖于置信度的定义。

### 四、Dempster 融合本身没有带来四分类增益

> **2026-10-13 补充**：本节的数字都来自 NLI 关系概率。换成大模型的关系概率后，
> 「D-S 不优于最好的 Statistic Gate」不变，但「两维判定对调充分的 conflict_aware
> 不显著」变成了显著，四个 Statistic Gate 之间也不再打平，见「十」。

「D-S 显著优于聚合类 baseline」里混着三件事：能输出 `conflicting`（输出空间）、
先分「证据不足 / 文档冲突」再看方向的**两维判定框架**、以及用 Dempster 融合
算这两个维度的分数。`conflict_aware` 只分离了第一件，而且它本身没有调充分。
两组补充对照把剩下两件拆开：

1. **Statistic Gate**（`src/rag_ds/baselines/stat_gate.py`）：判定规则、区域到
   类别的映射（与 D-S 共用 `gate_primary_state`）、网格构造与搜索协议都与 D-S
   相同，只把两个坐标轴换成简单统计量。共四个变体，**全部报告**：

   | 短名 | 横轴（证据不足） | 纵轴（文档冲突） |
   | --- | --- | --- |
   | `sr_std` | 平均 `p_unknown` | 各文档 `s - r` 的标准差 |
   | `max_product` | 平均 `p_unknown` | 最强支持 × 最强反驳 |
   | `sr_std_uprod` | `p_unknown` 连乘 | 各文档 `s - r` 的标准差 |
   | `max_product_uprod` | `p_unknown` 连乘 | 最强支持 × 最强反驳 |

   前两个是事先定好的；后两个（横轴换成连乘）是看到「人工标注输入的上限」
   结果之后补的，动机见该小节。补变体只会让对照更强，不会让 D-S 显得更好。
2. **conflict_aware 调充分**：判定阈值、冲突下限、平衡度（0.05–1.0）在内层
   三维网格上单独搜，不再固定 margin = 0.1、也不再与朴素 baseline 共用判定阈值。

```powershell
python scripts/run_cross_validation.py `
  --manifest data/processed/climate_fever_v2/manifest.json `
  --relations-template "outputs/ragchecker_v2/{split}_relations_{evaluator}_probs.jsonl" `
  --model-run-template "outputs/ragchecker_v2/{split}_model_run_{evaluator}_probs.json" `
  --evaluator refchecker_nli --single-evaluator refchecker_nli `
  --stat-gate sr_std max_product sr_std_uprod max_product_uprod `
  --conflict-margins 0.05 0.1 0.2 0.3 0.5 1.0 `
  --out-dir outputs/metrics/cross_validation_controls
```

两个选项都不给时输出与原协议**逐字节一致**（已用 `cmp` 核对
`outputs/metrics/cross_validation/` 下的两个文件）。

**结果**（v2，616 条，与主实验同一分折种子 42，阈值逐折在内层重选）：

| 方法 | Accuracy | Macro-F1 | 95% CI |
| --- | --- | --- | --- |
| Statistic Gate（`max_product_uprod`） | 0.3831 | 0.3659 | [0.3266, 0.4040] |
| **D-S** | 0.3750 | **0.3535** | [0.3145, 0.3932] |
| Statistic Gate（`max_product`） | 0.3734 | 0.3535 | [0.3153, 0.3920] |
| Statistic Gate（`sr_std`） | 0.3685 | 0.3519 | [0.3134, 0.3907] |
| Statistic Gate（`sr_std_uprod`） | 0.3620 | 0.3481 | [0.3094, 0.3863] |
| conflict_aware（调充分） | 0.3847 | 0.3374 | [0.3022, 0.3705] |
| conflict_aware（原协议） | 0.3653 | 0.3074 | [0.2738, 0.3419] |
| Weighted Average / Single Evaluator | 0.3442 | 0.2512 | [0.2242, 0.2773] |
| Majority Vote | 0.3425 | 0.2496 | [0.2229, 0.2759] |

| 配对比较（5000 次自助） | Δ Macro-F1 | 95% CI | p |
| --- | --- | --- | --- |
| D-S − Statistic Gate（`max_product`） | +0.0001 | [−0.0174, +0.0172] | 0.990 |
| D-S − Statistic Gate（`sr_std`） | +0.0016 | [−0.0282, +0.0306] | 0.917 |
| D-S − Statistic Gate（`sr_std_uprod`） | +0.0054 | [−0.0250, +0.0361] | 0.736 |
| D-S − Statistic Gate（`max_product_uprod`） | −0.0124 | [−0.0344, +0.0092] | 0.263 |
| D-S − conflict_aware（调充分） | +0.0161 | [−0.0236, +0.0565] | 0.410 |
| 四个 Statistic Gate − conflict_aware（调充分） | +0.011 ~ +0.029 | 均跨 0 | 0.15–0.62 |
| D-S − 三个朴素 baseline | +0.102 ~ +0.104 | 下界 ≥ +0.059 | < 0.001 |

conflict_aware 的提升**全部来自放宽 margin**：各折选出的都是判定阈值 0.35、
冲突下限 0.01、margin 0.5 或 1.0（1.0 即不要求两方向势均力敌）。只让它用自己
的判定阈值、margin 仍固定 0.1 时，Macro-F1 是 0.3070，与原协议的 0.3074 几乎
相同，D-S 对它仍是 +0.0465（p = 0.047）—— 原先那个「刚好显著」的差距，来自
固定 margin = 0.1 这个没搜过的选择（输出在
`outputs/metrics/cross_validation_ca_margin_fixed/`）。

**为什么会打平。** 本数据是单评估器、全部可靠性为 1.0，此时 Dempster 顺序
融合有一个精确恒等式：

```
            ∏ p_unknown(i)
m_theta = ─────────────────
             1 − K_doc
```

616 条上实测最大误差 5×10⁻¹⁵。`K_doc` 的中位数只有 0.005，所以对大多数
claim，`m_theta` 就是各文档 `p_unknown` 的连乘 —— 一个不需要证据理论也能
算出来的量。D-S 与 Statistic Gate（`max_product`）在 92.0% 的 claim 上给出
相同预测（`sr_std` 版 83.6%），各类 F1 也几乎相同。D-S 在形式上真正与简单
统计量不同的地方 —— 可靠性折扣、多评估器的可靠性加权 —— 在这份数据上
几乎用不上（可靠性全为 1.0，见「可靠性折扣的敏感性分析」）。

调充分的 conflict_aware 与 D-S 差距不显著，但**错误模式不同**：它在 616 条里
只判了 3 条 `supported`（F1 0.025），D-S 判了 51 条（F1 0.273）；它在
`refuted` / `insufficient` / `conflicting` 三类上 F1 反而更高。

**换分折种子，结论不变。** `scripts/run_cv_seed_robustness.py` 加上同样两个
选项，5 个种子全部报告（输出 `outputs/metrics/cv_seed_robustness_controls.csv`）：

| 对手 | Δ 均值 | Δ 标准差 | Δ 范围 | 显著种子数 |
| --- | --- | --- | --- | --- |
| Statistic Gate（`max_product`） | +0.0006 | 0.0056 | [−0.0076, +0.0082] | 0/5 |
| Statistic Gate（`sr_std`） | +0.0025 | 0.0043 | [−0.0056, +0.0061] | 0/5 |
| Statistic Gate（`max_product_uprod`） | −0.0008 | 0.0081 | [−0.0124, +0.0108] | 0/5 |
| Statistic Gate（`sr_std_uprod`） | +0.0029 | 0.0033 | [−0.0034, +0.0055] | 0/5 |
| conflict_aware（调充分） | +0.0198 | 0.0055 | [+0.0151, +0.0304] | 0/5 |
| 三个朴素 baseline | +0.108 ~ +0.109 | 0.006–0.007 | [+0.102, +0.121] | 5/5 |

D-S 对四个 Statistic Gate 的差距五个种子平均不到 0.003，对调充分的 conflict_aware
稳定在 +0.02 左右但**没有一个种子显著**。与「一」同样的措辞限制：五次 CV 共用
同一批 616 条样本，衡量的是划分敏感性，不是独立重复实验。

**多评估器设定下结论相同。** 同一条命令把 `--evaluator` 换成两个或三个评估器
（输出在 `cross_validation_2eval_controls/` 与 `cross_validation_3eval_controls/`）。
Statistic Gate 把所有 (文档, 评估器) 组合的关系一起统计，不区分 `K_doc` 与
`K_eval`。Macro-F1：

| 设定 | D-S | `max_product` | `sr_std` | `max_product_uprod` | `sr_std_uprod` | conflict_aware（调充分） | 朴素 baseline |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 单评估器（roberta） | 0.3535 | 0.3535 | 0.3519 | 0.3659 | 0.3481 | 0.3374 | 0.250–0.251 |
| 双评估器（+ deberta） | 0.3320 | 0.3448 | 0.3331 | 0.3321 | 0.3264 | 0.3363 | 0.245–0.253 |
| 三评估器（+ flan-t5） | 0.3302 | 0.3351 | 0.3276 | 0.3305 | 0.3242 | **0.3591** | 0.250–0.254 |

三种设定下 D-S 与四个 Statistic Gate、与调充分的 conflict_aware 的配对差异
**全部不显著**（p 在 0.14 到 0.99 之间）；D-S 对三个朴素 baseline 始终显著
（Δ = +0.076 ~ +0.104，p < 0.001）。三评估器下数值最高的是调充分的
conflict_aware（D-S − 它 = −0.029，CI [−0.067, +0.010]，p = 0.136）——
D-S 的多评估器融合没有比「平均后加一条冲突规则」更好地利用多个评估器。

**D-S 的可靠性折扣也没有独立价值。** 「二」里内层估计评估器可靠性后，D-S 的
多评估器退化被消除，那是可靠性折扣第一次在真实数据上起作用。但 Statistic Gate
拿到**同一份**估计出来的可靠性（作加权与折扣的权重）之后同样打平
（`scripts/run_reliability_estimation_cv.py --stat-gate ...`，输出在
`cross_validation_*_est_rel_controls/`）。Macro-F1：

| 评估器组合 | D-S | `max_product` | `sr_std` | `max_product_uprod` | `sr_std_uprod` |
| --- | --- | --- | --- | --- | --- |
| roberta + deberta | 0.3466 | 0.3425 | 0.3379 | 0.3478 | 0.3411 |
| roberta + flan-t5 | 0.3452 | 0.3202 | 0.3295 | 0.3247 | 0.3266 |
| 三者 | 0.3467 | 0.3349 | 0.3336 | 0.3349 | 0.3349 |

D-S − Statistic Gate 在 −0.001 到 +0.025 之间，12 组对比**全部不显著**
（p = 0.17–0.87）。roberta + flan-t5 下 D-S 数值上略高（+0.016 ~ +0.025），
但区间都跨 0。（这个脚本里的 conflict_aware 仍是原协议，没有搜 margin。）

**人工标注输入的上限：关系评估器才是瓶颈，证据不足一维要用连乘。** 把 NLI 的
关系概率换成数据集自带的人工投票分布（annotation oracle），其余协议不变
（输出在 `outputs/metrics/cross_validation_oracle_controls/`）：

```powershell
python scripts/run_cross_validation.py `
  --manifest data/processed/climate_fever_v2/manifest.json `
  --relations-template "data/processed/climate_fever_v2/{split}_relations.jsonl" `
  --evaluator climate_fever_human_vote_distribution `
  --single-evaluator climate_fever_human_vote_distribution `
  --stat-gate sr_std max_product sr_std_uprod max_product_uprod `
  --conflict-margins 0.05 0.1 0.2 0.3 0.5 1.0 `
  --out-dir outputs/metrics/cross_validation_oracle_controls
```

| 方法 | Accuracy | Macro-F1 | D-S − 它 | 95% CI | p |
| --- | --- | --- | --- | --- | --- |
| Statistic Gate（`max_product_uprod`） | 0.8409 | **0.8520** | −0.0232 | [−0.0439, −0.0030] | 0.023 |
| **D-S** | 0.8263 | 0.8288 | — | — | — |
| Statistic Gate（`sr_std_uprod`） | 0.8101 | 0.8172 | +0.0115 | [−0.0157, +0.0385] | 0.416 |
| Statistic Gate（`max_product`） | 0.7695 | 0.7789 | +0.0499 | [+0.0275, +0.0724] | < 0.001 |
| Statistic Gate（`sr_std`） | 0.7386 | 0.7425 | +0.0863 | [+0.0579, +0.1152] | < 0.001 |
| conflict_aware（调充分） | 0.7208 | 0.7201 | +0.1086 | [+0.0681, +0.1486] | < 0.001 |
| Weighted Average / Single Evaluator | 0.5795 | 0.4993 | +0.3295 | [+0.2923, +0.3652] | < 0.001 |
| Majority Vote | 0.4740 | 0.4098 | +0.4190 | [+0.3783, +0.4604] | < 0.001 |

> **这不是模型性能。** 金标准本身就是由这些人工投票按规则合成的（至少一篇
> 支持且其余不确定 → 支持；全部不确定 → 证据不足；既有支持又有反驳 → 冲突）。
> 这张表回答的只是：**关系输入足够准时，各种聚合方式能把标注规则还原到什么
> 程度**。不能与任何 NLI / LLM 评估器的结果并列报告。

三点读法：

1. **瓶颈在关系评估器。** 同一批方法，输入从 NLI 换成人工投票，Macro-F1 从
   0.35 左右升到 0.72–0.85。NLI 输入下「D-S / Statistic Gate / 调充分的
   conflict_aware 互相不显著」，是因为差异被评估器的噪声淹没了。
2. **两维判定框架在输入准确时显著优于其余方法。** 横轴用连乘的 Statistic Gate
   比调充分的 conflict_aware 高 +0.097 ~ +0.132（p < 0.001），D-S 比它高
   +0.109；对朴素 baseline 高 +0.33 以上。
3. **起作用的是「连乘」，不是 Dempster 规则。** 横轴用平均时 Statistic Gate
   显著低于 D-S（−0.050、−0.086）；换成连乘后一个与 D-S 打平、一个**显著
   高于** D-S（+0.023）。由上面的恒等式，`m_theta` 本来就是连乘除以
   `1 − K_doc`。连乘对应的语义是「**所有**文档都说不清才算证据不足，只要有
   一篇给出明确意见就不算」，这正是标注规则；平均会被说不清的文档稀释。
   NLI 输入下连乘与平均没有可见差别（四个变体全部打平）。

**能写进论文的：**

1. 显式区分「证据不足」与「文档冲突」的方法都远好于朴素聚合，这一条稳：
   D-S 对三个朴素 baseline +0.10，p < 0.001，5 个分折种子、三种评估器设定
   下都成立；
2. **Dempster 融合没有带来超过同一判定框架下简单统计量的四分类收益**：NLI
   输入下四个 Statistic Gate 变体与 D-S 全部打平（单 / 双 / 三评估器、估计
   可靠性、5 个种子）；人工标注输入下，横轴用连乘的变体与 D-S 打平或更好；
3. 两维判定框架对调充分的 conflict_aware 的优势，NLI 输入下**不显著**
   （Δ ≈ +0.01 ~ +0.03），人工标注输入下**显著**（+0.10 ~ +0.13）；
4. 「证据不足」这一维应当用连乘而不是平均（人工标注输入下差 0.05–0.09，
   显著）。

**不能再写的：**「D-S 融合机制显著优于 baseline」「+0.046 来自融合机制」
「对 conflict_aware 5/5 种子显著」「可靠性折扣是 D-S 独有的优势」。

D-S 若要在准确率上体现独立价值，需要**文档级**可靠性确有差异的数据（本数据
全为 1.0）—— 这是待做的实验，不是已有结论。

### 五、m_theta / K_doc 作为检测分数不优于简单统计量

> **2026-10-13 补充**：换成大模型的关系概率后，本节结论在 CLIMATE-FEVER 上不变；
> AVeriTeC 上 K_doc 成为 AUROC 最高的冲突分数，并显著高于「只数证据篇数」，
> 见「十」。

原实验 3 里 baseline 的冲突分数 `1 - |s - r|` 把「全都说不清」当成冲突，
AUROC 低于随机。这里把一组定义合理的简单统计量**事先全部列出、全部报告**
（定义见 `src/rag_ds/experiments/detection_proxies.py`），逐一与 D-S 分数做
配对自助 AUROC 比较。分数都与阈值无关，直接在全体 616 条上算：

```powershell
python scripts/run_detection_proxies.py `
  --manifest data/processed/climate_fever_v2/manifest.json `
  --relations-template "outputs/ragchecker_v2/{split}_relations_{evaluator}_probs.jsonl" `
  --model-run-template "outputs/ragchecker_v2/{split}_model_run_{evaluator}_probs.json" `
  --evaluator refchecker_nli --out-dir outputs/metrics/detection_proxies
```

**冲突识别**（正类 `conflicting`，154 条）：

| 分数 | AUROC | AUPRC | K_doc − 它 | 95% CI | p |
| --- | --- | --- | --- | --- | --- |
| **K_doc**（D-S） | 0.6063 | 0.3831 | — | — | — |
| `1 - \|s̄ - r̄\|`（原实验用的代理） | 0.4634 | 0.2229 | +0.1430 | [+0.0533, +0.2346] | 0.002 |
| `min(s̄, r̄)` | 0.5950 | 0.3681 | +0.0113 | [−0.0138, +0.0370] | 0.381 |
| `s̄ × r̄` | 0.5962 | 0.3741 | +0.0101 | [+0.0006, +0.0195] | 0.041 |
| 最强支持 × 最强反驳 | 0.6116 | 0.3827 | −0.0052 | [−0.0164, +0.0052] | 0.340 |
| `min(最强支持, 最强反驳)` | 0.5971 | 0.3777 | +0.0092 | [−0.0192, +0.0378] | 0.510 |
| 各文档 `s - r` 的标准差 | 0.6268 | 0.4031 | −0.0205 | [−0.0594, +0.0170] | 0.292 |
| 证据篇数（混杂因素对照） | 0.5000 | 0.2500 | +0.1063 | [+0.0532, +0.1606] | < 0.001 |

**证据不足识别**（正类 `insufficient`，154 条）：

| 分数 | AUROC | AUPRC | m_theta − 它 | 95% CI | p |
| --- | --- | --- | --- | --- | --- |
| **m_theta**（D-S） | 0.6213 | 0.3083 | — | — | — |
| 平均 `p_unknown` | 0.6144 | 0.3065 | +0.0069 | [−0.0008, +0.0150] | 0.076 |
| 最小 `p_unknown` | 0.6275 | 0.3137 | −0.0062 | [−0.0151, +0.0023] | 0.155 |
| `p_unknown` 连乘 | 0.6222 | 0.3087 | −0.0009 | [−0.0025, +0.0008] | 0.303 |

`K_doc` 只显著赢了原来那个过弱的代理，和 `s̄ × r̄`（CI 下界 +0.0006，边缘）；
与其余统计量打平，数值上最高的反而是 `s - r` 的标准差。`m_theta` 与三个简单
统计量全部打平，与 `p_unknown` 连乘只差 0.001 —— 由上面的恒等式可知这是必然的。
**检测分数本身不是 D-S 的优势**；原实验 3「唯一站得住的优势」一说撤回。

「证据篇数」一行是混杂因素对照：CLIMATE-FEVER 每条 claim 固定 5 篇，篇数没有
任何信息（AUROC 恒为 0.5）。证据篇数不固定的数据集上情况完全不同，见「八」。

### 六、v2 嵌套消融：两个判定维度各自显著有用

原消融只在 v1 的 80 条上做过，离散版与连续版对 theta 门控结论相反。这里用与
主实验相同的分折、种子与网格，每个变体在每折内层**各自重搜剩下那一维**的
阈值（沿用完整方法的阈值会低估消融变体），616 条汇总后与 `full` 配对检验：

```powershell
python scripts/run_cv_ablation.py `
  --manifest data/processed/climate_fever_v2/manifest.json `
  --relations-template "outputs/ragchecker_v2/{split}_relations_{evaluator}_probs.jsonl" `
  --model-run-template "outputs/ragchecker_v2/{split}_model_run_{evaluator}_probs.json" `
  --evaluator refchecker_nli --out-dir outputs/metrics/cross_validation_ablation
```

| 变体 | Accuracy | Macro-F1 | Δ vs full | 95% CI | p |
| --- | --- | --- | --- | --- | --- |
| full | 0.3750 | 0.3535 | — | — | — |
| no_theta_gate | 0.3360 | 0.2725 | **−0.0810** | [−0.1068, −0.0547] | < 0.001 |
| no_doc_conflict_gate | 0.3636 | 0.2999 | **−0.0537** | [−0.0854, −0.0222] | 0.001 |
| no_two_dimensional_gate | 0.3247 | 0.2146 | **−0.1389** | [−0.1770, −0.1011] | < 0.001 |

`full` 与主实验的 D-S 逐条一致、各折阈值相同（有测试钉住），所以这里的 Δ 就是
相对论文主表那个方法的变化。`no_reliability` 没有跑：数据集可靠性全为 1.0，
去折扣是恒等变换。

三组对照合起来的读法：**两个判定维度都必不可少（本节），但用 D-S 还是用简单
统计量来算这两个维度，在这份数据上没有区别（「四」「五」）。** 论文的贡献应
落在「证据不足 / 文档冲突」两维证据状态诊断这个框架上，D-S 是它的一种有原则、
可解释的实现，而不是准确率来源。

### 七、选择性回答：对新对照只剩一个不稳定的趋势

「三」里 D-S 的置信度在 `class_conditional` 口径下显著优于朴素 baseline。同一
分析换成补充对照的结果重做（Statistic Gate 的置信度与 D-S 同口径：判哪一类
就取那一类对应的平均分、横轴统计量或冲突统计量）：

```powershell
python scripts/selective_answer_analysis.py --profile controls
```

输出在 `outputs/metrics/selective_answer_controls/`。`class_conditional` 口径下
D-S 减去对手（2000 次配对自助，`*` = 95% 区间不跨 0；ΔAUROC > 0、ΔAURC < 0
表示 D-S 占优）：

| 设定 | 对手 | ΔAUROC | 95% CI | ΔAURC | 95% CI |
| --- | --- | --- | --- | --- | --- |
| 单评估器 | Statistic Gate（`max_product`） | +0.036 | [−0.000, +0.075] | +0.005 | [−0.009, +0.020] |
| | Statistic Gate（`sr_std`） | +0.043 | [−0.003, +0.088] | +0.000 | [−0.029, +0.029] |
| | Statistic Gate（`max_product_uprod`） | +0.039\* | [+0.002, +0.079] | +0.010 | [−0.007, +0.027] |
| | Statistic Gate（`sr_std_uprod`） | +0.036 | [−0.009, +0.079] | −0.014 | [−0.046, +0.018] |
| | conflict_aware（调充分） | **+0.118\*** | [+0.065, +0.170] | −0.019 | [−0.058, +0.022] |
| | weighted_average | **+0.098\*** | [+0.028, +0.163] | **−0.054\*** | [−0.096, −0.013] |
| 双评估器 | Statistic Gate（`max_product`） | +0.061\* | [+0.001, +0.122] | +0.013 | [−0.017, +0.042] |
| | Statistic Gate（`sr_std`） | +0.055 | [−0.007, +0.118] | −0.011 | [−0.048, +0.028] |
| | Statistic Gate（`max_product_uprod`） | +0.004 | [−0.049, +0.057] | +0.007 | [−0.018, +0.032] |
| | Statistic Gate（`sr_std_uprod`） | +0.013 | [−0.046, +0.070] | −0.012 | [−0.050, +0.021] |
| | conflict_aware（调充分） | +0.045 | [−0.013, +0.106] | +0.028 | [−0.017, +0.073] |
| | weighted_average | **+0.093\*** | [+0.018, +0.163] | −0.023 | [−0.072, +0.022] |

- 单评估器下，D-S 置信度对「判错」的排序能力（AUROC）对四个 Statistic Gate
  **方向一致**（+0.036 ~ +0.043），但区间下界都贴着 0。两个带 `*` 的下界只有
  +0.001 ~ +0.002：只跑两个变体时同一组对比的下界是 −0.002 与 −0.000，
  换一条重采样随机序列「显著」就翻转，**不能当作显著**。
- 双评估器下，对横轴用连乘的两个变体已经打平（+0.004、+0.013）。
- 弃答后的准确率收益（AURC）对四个 Statistic Gate 全部不显著。
- `max_signal` 口径下 D-S 对 Statistic Gate 没有优势，双评估器下对
  `max_product` 版显著更差（ΔAUROC = −0.030）。
- Statistic Gate 的置信度定义是这次补充实验里新定的（没有先例可循），换一种
  定义可能改变这组数字。

这是目前唯一能看到 D-S 可能有独立价值的地方（融合后的质量作为置信度，排序
判错的 claim 略好），但只是**不稳定的趋势**，论文里只能写成「值得在更大数据
上检验的观察」，不能写成结论。

### 八、第二个数据集 AVeriTeC：结论可以推广，但必须先控制「证据篇数」

> **2026-10-13 补充**：本节用的是 NLI 关系概率。换成大模型后，下面三条结论里的
> 第 3 条变了：控制证据篇数之后，两维判定对调充分的 conflict_aware 仍然显著更好，
> 见「十」。

CLIMATE-FEVER 的 claim 标签由逐条证据的多数票按规则合成，两维判定在结构上
复刻了这条规则。AVeriTeC（Schlichtkrull et al., NeurIPS 2023 D&B）的四类判定
由事实核查员直接给出，用来检验「两维判定有效是不是只因为贴合了造标签的方式」。

**数据。** 官方 train + dev（CC-BY-NC-4.0），去重后四类各取 230 条，共 920 条
（`data/processed/averitec_v1/`，`src/rag_ds/datasets/averitec.py`）。与
CLIMATE-FEVER 有三处不同：

| | CLIMATE-FEVER v2 | AVeriTeC v1 |
| --- | --- | --- |
| 证据 | Wikipedia 句子 | 问答对，每个 (问题, 答案) 当一篇文档；「没查到」的答案保留 |
| 每条 claim 的证据篇数 | 固定 5 篇 | 1–16 篇，平均约 3 篇 |
| 逐条证据的立场标注 | 有（annotation oracle） | **没有**，清单里 `relation_predictions_kind` 为 `none` |

另外 `Conflicting Evidence/Cherrypicking` 把「断章取义」也算在内，并不总是文档
互相矛盾。

```powershell
# 原始文件（固定到一个仓库版本，地址见 rag_ds.datasets.averitec.AVERITEC_DOWNLOAD_URLS）
curl -L -o data/raw/averitec/train.json <train 的地址>
curl -L -o data/raw/averitec/dev.json   <dev 的地址>
python scripts/prepare_averitec.py

# 每个 split：导出 → NLI（与 CLIMATE-FEVER 同一个模型）→ 摄取
python scripts/export_ragchecker_input.py --manifest data/processed/averitec_v1/manifest.json `
  --split test --out outputs/ragchecker_averitec/test_checking_inputs.json
.venv-refchecker\Scripts\python.exe scripts/run_refchecker_nli_chunked.py `
  --input outputs/ragchecker_averitec/test_checking_inputs.json `
  --output outputs/ragchecker_averitec/test_checking_outputs_probs.json --emit-probabilities
python scripts/ingest_ragchecker.py --manifest data/processed/averitec_v1/manifest.json --split test `
  --outputs outputs/ragchecker_averitec/test_checking_outputs_probs.json --use-probabilities `
  --evaluator refchecker_nli --checker-name ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli `
  --out-dir outputs/ragchecker_averitec
```

2790 个 (claim, 文档) 组合，NLI 标签分布：Neutral 约 78%、Contradiction 约 13%、
Entailment 约 8.5%（CLIMATE-FEVER 上 Entailment 只有 3.5%）。之后的交叉验证、
消融、检测与种子稳健性都是「四」「五」「六」里的同一批脚本，只换 `--manifest`
与两个路径模板，输出在 `outputs/metrics/averitec_*`。

**先看不加控制的结果**（920 条，5 折嵌套交叉验证）：

| 方法 | Accuracy | Macro-F1 | D-S − 它 | 95% CI | p |
| --- | --- | --- | --- | --- | --- |
| Statistic Gate（`max_product`） | 0.4185 | 0.4020 | −0.0077 | [−0.0312, +0.0162] | 0.527 |
| Statistic Gate（`max_product_uprod`） | 0.4120 | 0.3945 | −0.0003 | [−0.0225, +0.0221] | 0.972 |
| **D-S** | 0.4076 | **0.3942** | — | — | — |
| Statistic Gate（`sr_std`） | 0.3891 | 0.3808 | +0.0134 | [−0.0174, +0.0447] | 0.396 |
| Statistic Gate（`sr_std_uprod`） | 0.3804 | 0.3716 | +0.0226 | [−0.0076, +0.0527] | 0.146 |
| conflict_aware（调充分） | 0.3533 | 0.3373 | **+0.0570** | [+0.0249, +0.0884] | 0.001 |
| Weighted Average / Single Evaluator | 0.3500 | 0.2716 | +0.1227 | [+0.0902, +0.1545] | < 0.001 |
| Majority Vote | 0.3402 | 0.2571 | +0.1371 | [+0.1046, +0.1698] | < 0.001 |

5 个分折种子下 D-S 对调充分的 conflict_aware 全部显著（Δ 均值 +0.0537），对四个
Statistic Gate 全部不显著。嵌套消融：

| 变体 | Macro-F1 | Δ vs full | 95% CI | p |
| --- | --- | --- | --- | --- |
| full | 0.3942 | — | — | — |
| no_theta_gate | 0.3307 | **−0.0635** | [−0.0914, −0.0357] | < 0.001 |
| no_doc_conflict_gate | 0.3270 | **−0.0672** | [−0.0921, −0.0409] | < 0.001 |
| no_two_dimensional_gate | 0.2445 | **−0.1498** | [−0.1836, −0.1154] | < 0.001 |

到这里看上去比 CLIMATE-FEVER 更好：两维判定显著优于调充分的 conflict_aware。
**但这一条是假象。**

**混杂因素：证据篇数。** AVeriTeC 里冲突类 claim 平均有 3.95 篇证据，其余三类
只有 2.6–3.0 篇（标注员为有争议的 claim 找了更多证据）。而 `K_doc = 1 − ∏(1 − K_i)`
会随文档数累积变大。检测实验（`outputs/metrics/averitec_detection_proxies/`）：

| 识别冲突的分数 | AUROC | K_doc − 它 | 95% CI |
| --- | --- | --- | --- |
| **证据篇数本身**（`document_count`） | **0.7141** | −0.0311 | [−0.0688, +0.0068] |
| K_doc（D-S） | 0.6830 | — | — |
| 各文档 `s − r` 的标准差 | 0.6681 | +0.0149 | [−0.0056, +0.0360] |
| 最强支持 × 最强反驳 | 0.6189 | +0.0640 | [+0.0382, +0.0887] |
| `min(最强支持, 最强反驳)` | 0.6075 | +0.0755 | [+0.0429, +0.1079] |
| `s̄ × r̄` | 0.5794 | +0.1036 | [+0.0732, +0.1319] |
| `min(s̄, r̄)` | 0.5761 | +0.1068 | [+0.0707, +0.1423] |

`K_doc` 在这里显著赢了大多数简单统计量（CLIMATE-FEVER 上是打平），但**完全不看
文档内容、只数有几篇**就有 0.714，比它还高。在篇数相同的 claim 内部重新比
（`conflict_auroc_by_document_count`），`K_doc` 的领先就消失了：

| 证据篇数 | claim | 其中冲突 | K_doc | `s − r` 标准差 | 最强支持 × 最强反驳 |
| --- | --- | --- | --- | --- | --- |
| 1 | 157 | 11 | 0.500 | 0.500 | 0.506 |
| 2 | 285 | 40 | 0.584 | 0.565 | 0.576 |
| 3 | 200 | 57 | 0.623 | 0.656 | 0.636 |
| 4 | 128 | 50 | 0.552 | 0.524 | 0.540 |
| 5 | 63 | 31 | 0.579 | 0.627 | 0.601 |

四分类上同样如此。把 Statistic Gate 的纵轴直接换成「证据篇数」（`--stat-gate
doc_count`，混杂因素对照，不是冲突统计量；输出在
`outputs/metrics/averitec_cross_validation_doc_count/`）：

| 方法 | Macro-F1 | D-S − 它 | 95% CI | p |
| --- | --- | --- | --- | --- |
| **只数篇数**（`doc_count`） | **0.4387** | −0.0445 | [−0.0748, −0.0141] | 0.005 |
| 只数篇数 + 连乘（`doc_count_uprod`） | 0.4332 | −0.0389 | [−0.0687, −0.0091] | 0.010 |
| D-S | 0.3942 | — | — | — |

各折选出的规则都是「证据 ≥ 3 篇就判冲突」。它**显著高于 D-S**。也就是说，上面
「两维判定显著优于 conflict_aware」「去掉冲突门控掉 0.067」里，冲突这一维的
贡献主要来自数据集的这个特点，不能当作识别冲突的证据。「证据不足」一维不受
影响：篇数对识别证据不足没有信息（AUROC 0.514），`m_theta` 与三个简单统计量
打平（0.645 对 0.642–0.645）。

**控制篇数之后。** `--first-documents K`：只保留证据至少 K 篇的 claim，每条只用
原始顺序里的前 K 篇，于是所有 claim 篇数相同（不需要重新推理；输出在
`outputs/metrics/averitec_cross_validation_first2/` 与 `…_first3/`）：

| | K = 2（763 条） | K = 3（478 条） |
| --- | --- | --- |
| D-S Macro-F1 | 0.2956 | 0.3362 |
| 四个 Statistic Gate | 0.298–0.308 | 0.328–0.356 |
| conflict_aware（调充分） | 0.2781 | 0.2964 |
| 朴素 baseline | 0.180–0.214 | 0.198–0.206 |
| D-S − Statistic Gate | −0.013 ~ −0.003，均不显著 | −0.020 ~ +0.008，均不显著 |
| D-S − conflict_aware（调充分） | +0.0175，CI [−0.020, +0.054]，p = 0.372 | +0.0399，CI [−0.009, +0.089]，p = 0.109 |
| D-S − 朴素 baseline | +0.082 ~ +0.116，p < 0.001 | +0.130 ~ +0.138，p < 0.001 |

篇数控制住之后，AVeriTeC 与 CLIMATE-FEVER 的模式**完全一致**：

1. 显式区分「证据不足」与「文档冲突」的方法都显著优于朴素聚合 —— 在标签由人工
   判定、而不是规则合成的数据上同样成立；
2. D-S 与同框架的简单统计量打平；
3. 两维判定对调充分的 conflict_aware 的优势不显著。

**必须写进论文的两点。** 第一，证据篇数不固定的数据集上，任何会随篇数累积的
冲突分数都要与「只数篇数」对照，否则会把数据集的采集特点当成方法的能力。
第二，截断到前 K 篇会丢证据（冲突的那一篇可能排在后面），K = 2 / K = 3 的绝对
分数因此偏低，它们是**对照**，不是 AVeriTeC 上的最好成绩。

### 九、CONFACT：文档可靠性确有差异，折扣仍然没有显著效果

前面所有数据的文档可靠性都是 1.0。CONFACT（Ge et al., IJCAI 2025）的 HumC 子集
给每条 claim 配了约 12 篇互相冲突的网页，并附带 Media Bias/Fact Check 对 6100
家媒体的可信度分级，用来检验 D-S 的**文档级**可靠性折扣。

**数据与处理**（`src/rag_ds/datasets/confact.py`）：

- 287 条 claim，标签只有两类（Refuted 236、Supported 51），所以检验的是**倾向
  判断**，不是四类证据状态；
- 原始数据是 pickle。`pickle.load` 会执行文件里指定的任意可调用对象，这里用
  拒绝导入任何类的 Unpickler 读取，只接受内置容器、字符串与数字；
- 网页中位数约 940 词，NLI 一次只能看 512 个词元：每篇网页切成 120 词的重叠
  窗口，取与 claim 的 TF-IDF 余弦相似度最高的一段。3464 篇里 872 篇内容为空，
  余下 2592 篇；
- 可信度 → 可靠性用**事先定好**的映射：high 1.0、medium 0.6、low 0.2，没有
  评级的 0.6。2592 篇里 high 765、medium 409、low 89、无评级 1329；
- 上游仓库没有声明许可证：原始文件、处理后的样本与含网页段落的 NLI 输入输出
  **都不入库**，只有不含原文的指标在 `outputs/metrics/confact_reliability/`。

```powershell
python scripts/prepare_confact.py
python scripts/run_confact_reliability.py export
.venv-refchecker\Scripts\python.exe scripts/run_refchecker_nli_chunked.py `
  --input outputs/ragchecker_confact/checking_inputs.json `
  --output outputs/ragchecker_confact/checking_outputs_probs.json --emit-probabilities
python scripts/run_confact_reliability.py analyse
```

倾向只看支持与反驳哪边大（D-S 用 `m_support − m_refute`，加权平均用加权后的
`s̄ − r̄`），没有要调的阈值，直接在 287 条上评估。AUROC 把这个差当连续分数，
不受两类比例影响：

| 文档可靠性 | 方法 | Accuracy | Macro-F1 | AUROC | 未定 |
| --- | --- | --- | --- | --- | --- |
| 全为 1 | D-S | 0.5052 | 0.4567 | 0.5289 | 2 |
| 全为 1 | 加权平均 | 0.5192 | 0.4600 | 0.5255 | 0 |
| 按可信度 | D-S | 0.5331 | 0.4791 | 0.5523 | 1 |
| 按可信度 | 加权平均 | 0.5261 | 0.4757 | 0.5349 | 0 |
| — | 多数票 | 0.4111 | 0.4006 | 0.5169 | 46 |

| 比较（5000 次配对自助） | Δ Macro-F1 | 95% CI | Δ AUROC | 95% CI |
| --- | --- | --- | --- | --- |
| D-S：按可信度 − 全为 1 | +0.0224 | [−0.010, +0.057] | +0.0234 | [−0.023, +0.069] |
| 加权平均：按可信度 − 全为 1 | +0.0157 | [−0.006, +0.041] | +0.0094 | [−0.004, +0.023] |
| 按可信度：D-S − 加权平均 | +0.0034 | [−0.018, +0.025] | +0.0174 | [−0.021, +0.055] |
| 全为 1：D-S − 加权平均 | −0.0033 | [−0.028, +0.024] | +0.0034 | [−0.018, +0.024] |

敏感性扫描（low ∈ {0, 0.2, 0.4} × 无评级 ∈ {0.4, 0.6, 0.8, 1.0}，共 12 组，全部
在 `sensitivity.csv`）：D-S 的 Macro-F1 在 0.436–0.492、AUROC 在 0.507–0.561
之间；加权平均在 0.452–0.483 与 0.520–0.545 之间。

**结论，以及它为什么只是弱证据：**

1. 加入可靠性后两个方法的数字都往好的方向动（D-S +0.022、加权平均 +0.016），
   但**都不显著**；
2. 同一份可靠性下，D-S 的折扣与把可靠性当权重的加权平均**打平**（+0.003）；
3. 所有方法的 AUROC 只有 0.52–0.56，接近随机。NLI 在「整页网页里挑出的一段」
   上基本判断不出倾向，可靠性再怎么加权也救不回来 —— 这个实验**检验不出**
   折扣有没有用，不能读成「折扣没用」；
4. 低可信度的文档只占 3.4%（89 篇），可靠性的差异本身就小。

**D-S 在高冲突下的数值问题在这里第一次真实出现。** 不做折扣时 `K_doc` 的中位数
是 0.63，287 条里 82 条超过 0.99、11 条超过 0.999999（NLI 的输出很极端：139 个
组合 `p_support > 0.999`，115 个 `p_refute > 0.999`）。其中 2 条近乎完全冲突，
Dempster 归一化的分母接近 0，`ds` 子包按设计报错而不是返回不可靠的数；实验里
记为 `undetermined`（算判错）并在 `ds_undefined` 列标出。按可信度折扣后 `K_doc`
中位数降到 0.30，这样的 claim 减到 1 条 —— 折扣确实缓解了高冲突，只是没有
转化成显著的准确率收益。

### 十、换成大模型：直接判四类，以及给两维判定当关系评估器

前面九节的关系概率都来自 NLI 小模型，「瓶颈在关系评估器」一直只有人工标注的
上限分析作证据。2026-10-12 至 10-13 用 DeepSeek V4.1 Flash（接口里的模型名
`deepseek-flash`，思考默认开启、JSON 输出模式、温度等其余参数都用默认值）补了
两组实验，同一个模型、两种用法：

- **直接判四类**（`scripts/run_llm_judge.py`）：把 claim 和它的全部证据一次性
  交给模型，直接输出四类之一。零样本，没有任何阈值，不看标签。这是审稿人最
  可能要的强对照。
- **当关系评估器**（`scripts/run_llm_relations.py`）：与 NLI 路径完全同构 ——
  每个（claim，文档）组合单独发一个请求，模型**一次只看一篇文档**，写出
  `entailment / neutral / contradiction` 三个概率。输出与 NLI 推理脚本同形，
  用 `ingest_ragchecker.py --use-probabilities` 摄取后，D-S、Statistic Gate、
  conflict_aware、朴素聚合、嵌套交叉验证、消融、检测、种子稳健性全部沿用
  「四」到「八」的同一批脚本，一个字不改。

```powershell
# 关系评估器：每个 split 一次；中断后原样重跑即可续跑（进度在 <output>.pairs.jsonl）
python scripts/run_llm_relations.py --concurrency 10 `
  --input outputs/ragchecker_v2/test_checking_inputs.json `
  --output outputs/ragchecker_v2/test_checking_outputs_deepseek_flash.json
python scripts/ingest_ragchecker.py --manifest data/processed/climate_fever_v2/manifest.json `
  --split test --outputs outputs/ragchecker_v2/test_checking_outputs_deepseek_flash.json `
  --use-probabilities --evaluator deepseek_flash --checker-name deepseek-flash `
  --out-dir outputs/ragchecker_v2
python scripts/run_cross_validation.py `
  --manifest data/processed/climate_fever_v2/manifest.json `
  --relations-template "outputs/ragchecker_v2/{split}_relations_{evaluator}_probs.jsonl" `
  --model-run-template "outputs/ragchecker_v2/{split}_model_run_{evaluator}_probs.json" `
  --evaluator deepseek_flash --single-evaluator deepseek_flash `
  --stat-gate sr_std max_product sr_std_uprod max_product_uprod `
  --conflict-margins 0.05 0.1 0.2 0.3 0.5 1.0 `
  --out-dir outputs/metrics/cross_validation_deepseek_flash_controls

# 直接判四类；--summary-name 让同一批结果再与另一份交叉验证预测配对，不重新调用 API
python scripts/run_llm_judge.py --provider deepseek --model deepseek-flash --max-tokens 16000 `
  --manifest data/processed/climate_fever_v2/manifest.json `
  --out-dir outputs/llm_judge/climate_fever_v2/deepseek-flash `
  --compare-with outputs/metrics/cross_validation_deepseek_flash_controls/cv_predictions.csv `
  --summary-name judge_summary_vs_deepseek_flash.json
```

密钥读自环境变量 `DEEPSEEK_API_KEY`，没有时读项目根目录 `.env` 里的同名行
（`.env` 在 `.gitignore` 里）。脚本只在发请求时使用密钥，不打印、不写进任何输出。

**规模与开销。** 关系评估 5870 个组合（CLIMATE-FEVER 3080、AVeriTeC 2790），
输入 179 万、输出 320 万词元；判四类 1536 条 claim，输入 78 万、输出 208 万词元
（输出含思考）。实际作答的模型全部是 `deepseek-flash`，没有拒答。关系评估的
请求数是判四类的 3.8 倍，按标价折算的开销约 1.6 倍。

输出位置：关系文件与谱系在 `outputs/ragchecker_v2/`、`outputs/ragchecker_averitec/`
下带 `deepseek_flash` 的文件；指标在 `outputs/metrics/*deepseek_flash*`；判四类在
`outputs/llm_judge/<数据集>/deepseek-flash/`。

#### 关系这一层：argmax 几乎没变，概率的排序好了很多

CLIMATE-FEVER 带人工逐篇标注。在人工多数票唯一的 2142 个组合上（支持 663、
反驳 581、不确定 898）比较两个评估器
（`scripts/compare_relation_inputs.py --with-oracle`，输出
`outputs/metrics/relation_input_comparison/summary.json`）：

| | NLI（RoBERTa-large） | 大模型 | 大模型 − NLI | 95% CI |
| --- | --- | --- | --- | --- |
| argmax 准确率 | 0.5299 | 0.5546 | — | — |
| AUROC：支持 | 0.6468 | 0.8129 | +0.1661 | [+0.1384, +0.1943] |
| AUROC：反驳 | 0.7696 | 0.7898 | +0.0202 | [−0.0070, +0.0460] |
| AUROC：不确定 | 0.5947 | 0.7805 | +0.1858 | [+0.1597, +0.2121] |

大模型比 NLI 更保守：全部 3080 个组合里 85% 判成 Neutral（NLI 是 77%），判成
Contradiction 的从 19% 降到 11%，Entailment 都不到 4%；两者 argmax 的一致率 81%，
只看「取最大的那一类」几乎看不出差别。差别在概率的**排序**上：一篇人工
判为支持的文档，大模型即使仍把它归为 Neutral，给它的支持概率也明显高于别的
文档。D-S 与各个统计量用的正是这些连续值，所以下游的变化比 argmax 准确率
暗示的大得多。**只报 argmax 准确率会严重低估评估器的差别。**

大模型的概率是**口头写出**的（谱系里记为 `probability_source: verbalized`），
比 softmax 粗：CLIMATE-FEVER 的 3080 个组合只有 167 种不同的三元组，AVeriTeC 的
2790 个组合只有 201 种。

#### 四分类：两维判定整体大幅上升，对 conflict_aware 的优势变得显著

5 折嵌套交叉验证，分折种子 42，协议与「四」「八」逐项相同，只换关系输入。
「NLI 时」一列是同一方法在 NLI 关系概率下的 Macro-F1。

**CLIMATE-FEVER v2（616 条）**

| 方法 | Accuracy | Macro-F1 | NLI 时 | D-S − 它 | 95% CI | p |
| --- | --- | --- | --- | --- | --- | --- |
| Statistic Gate（`max_product`） | 0.5341 | **0.5276** | 0.3535 | −0.0223 | [−0.0436, −0.0013] | 0.038 |
| Statistic Gate（`max_product_uprod`） | 0.5211 | 0.5187 | 0.3659 | −0.0134 | [−0.0311, +0.0039] | 0.130 |
| **D-S** | 0.5097 | 0.5053 | 0.3535 | — | — | — |
| Statistic Gate（`sr_std`） | 0.4773 | 0.4618 | 0.3519 | +0.0435 | [+0.0111, +0.0755] | 0.008 |
| 大模型直接判四类（零样本） | 0.4789 | 0.4555 | — | +0.0498 | [+0.0082, +0.0901] | 0.020 |
| Statistic Gate（`sr_std_uprod`） | 0.4643 | 0.4536 | 0.3481 | +0.0517 | [+0.0214, +0.0816] | < 0.001 |
| conflict_aware（调充分） | 0.3685 | 0.3187 | 0.3374 | **+0.1866** | [+0.1425, +0.2312] | < 0.001 |
| Majority Vote | 0.3149 | 0.2171 | 0.2496 | +0.2882 | [+0.2419, +0.3329] | < 0.001 |
| Weighted Average / Single Evaluator | 0.3149 | 0.2153 | 0.2512 | +0.2900 | [+0.2431, +0.3354] | < 0.001 |

**AVeriTeC（920 条）**

| 方法 | Accuracy | Macro-F1 | NLI 时 | D-S − 它 | 95% CI | p |
| --- | --- | --- | --- | --- | --- | --- |
| Statistic Gate（`max_product`） | 0.6065 | **0.5987** | 0.4020 | −0.0206 | [−0.0424, +0.0013] | 0.064 |
| Statistic Gate（`max_product_uprod`） | 0.5957 | 0.5885 | 0.3945 | −0.0105 | [−0.0261, +0.0049] | 0.174 |
| 大模型直接判四类（零样本） | 0.6207 | 0.5861 | — | −0.0081 | [−0.0399, +0.0214] | 0.610 |
| **D-S** | 0.5848 | 0.5781 | 0.3942 | — | — | — |
| Statistic Gate（`sr_std`） | 0.5087 | 0.5142 | 0.3808 | +0.0639 | [+0.0293, +0.0984] | 0.001 |
| Statistic Gate（`sr_std_uprod`） | 0.4913 | 0.4868 | 0.3716 | +0.0913 | [+0.0640, +0.1198] | < 0.001 |
| conflict_aware（调充分） | 0.4772 | 0.4733 | 0.3373 | **+0.1047** | [+0.0693, +0.1400] | < 0.001 |
| Weighted Average / Single Evaluator | 0.4261 | 0.3653 | 0.2716 | +0.2128 | [+0.1798, +0.2449] | < 0.001 |
| Majority Vote | 0.3935 | 0.3299 | 0.2571 | +0.2482 | [+0.2135, +0.2819] | < 0.001 |

同一个方法换关系输入前后的逐条配对比较
（`scripts/compare_relation_inputs.py`，输出在 `outputs/metrics/relation_input_comparison/`
与 `outputs/metrics/averitec_relation_input_comparison/`）：

| 大模型 − NLI（Δ Macro-F1） | CLIMATE-FEVER | 95% CI | AVeriTeC | 95% CI |
| --- | --- | --- | --- | --- |
| D-S | +0.1518 | [+0.0996, +0.2011] | +0.1838 | [+0.1457, +0.2229] |
| Statistic Gate（`max_product`） | +0.1741 | [+0.1223, +0.2238] | +0.1967 | [+0.1587, +0.2351] |
| conflict_aware（调充分） | −0.0187 | [−0.0601, +0.0206] | +0.1361 | [+0.1008, +0.1710] |
| Weighted Average | −0.0359 | [−0.0661, −0.0063] | +0.0937 | [+0.0694, +0.1198] |

评估器变强，受益的是两维判定：四个 Statistic Gate 与 D-S 在两个数据集上都显著上升
（+0.11 ~ +0.20）。聚合类方法在 AVeriTeC 上也显著上升（+0.07 ~ +0.14），在
CLIMATE-FEVER 上却没有受益 —— 原因见下面「conflict_aware 为什么不升反降」。

5 个分折种子，**全部报告**（`outputs/metrics/cv_seed_robustness_deepseek_flash_controls.csv`
与 `averitec_cv_seed_robustness_deepseek_flash_controls.csv`）。D-S 的 Macro-F1 在
CLIMATE-FEVER 上是 0.505–0.524，在 AVeriTeC 上是 0.568–0.587：

| D-S − 对手 | CLIMATE-FEVER：Δ 均值 | Δ 范围 | 显著 | AVeriTeC：Δ 均值 | Δ 范围 | 显著 |
| --- | --- | --- | --- | --- | --- | --- |
| Statistic Gate（`max_product`） | −0.0171 | [−0.0223, −0.0098] | 1/5 | −0.0243 | [−0.0342, −0.0153] | 2/5 |
| Statistic Gate（`max_product_uprod`） | −0.0136 | [−0.0218, −0.0060] | 2/5 | −0.0111 | [−0.0178, −0.0042] | 1/5 |
| Statistic Gate（`sr_std`） | +0.0534 | [+0.0435, +0.0650] | 5/5 | +0.0658 | [+0.0579, +0.0724] | 5/5 |
| Statistic Gate（`sr_std_uprod`） | +0.0553 | [+0.0469, +0.0684] | 5/5 | +0.0923 | [+0.0854, +0.0993] | 5/5 |
| conflict_aware（调充分） | +0.1864 | [+0.1807, +0.1938] | 5/5 | +0.1076 | [+0.0982, +0.1172] | 5/5 |
| Weighted Average / Single Evaluator | +0.2979 | [+0.2895, +0.3085] | 5/5 | +0.2140 | [+0.2022, +0.2248] | 5/5 |
| Majority Vote | +0.2967 | [+0.2882, +0.3067] | 5/5 | +0.2488 | [+0.2379, +0.2568] | 5/5 |

四点读法：

1. **瓶颈确实在关系评估器。** 下游一个字不改，只把 NLI 换成大模型，D-S 的
   Macro-F1 在两个数据集上分别涨了约 0.15 和 0.18；CLIMATE-FEVER 上离人工标注
   输入的上限（0.829）仍然很远。
2. **两维判定对调充分的 conflict_aware 的优势，从不显著变成又大又稳。** NLI 输入
   时 D-S 对它的差距在 CLIMATE-FEVER 上是 +0.016（p = 0.41，0/5 个种子显著），
   在 AVeriTeC 上控制篇数后是 +0.018（p = 0.37）；换成大模型后是 +0.19 与 +0.10，
   两个数据集、5 个种子全部显著。不用 D-S 的 Statistic Gate 也一样：CLIMATE-FEVER
   上四个变体全部显著高于 conflict_aware（+0.14 ~ +0.21）；AVeriTeC 上三个显著
   （+0.04 ~ +0.13），`sr_std_uprod` 不显著（+0.013，p = 0.48）。
3. **D-S 仍然不是两维框架里最好的实现。** `max_product` 两个变体在两个数据集、
   全部 5 个种子上的点估计都高于 D-S（均值差 0.011–0.024），各有 1–2 个种子显著。
   「四」的结论 —— 功劳属于两维判定框架，不属于 Dempster 规则 —— 不变。
4. **冲突一维用哪个统计量开始有区别了。** NLI 输入时四个 Statistic Gate 与 D-S
   互相打平；现在 D-S（`K_doc`）比标准差型的两个 `sr_std` 变体高 0.05–0.09，
   5 个种子全部显著，乘积型的 `max_product` 又略高于 D-S。

**conflict_aware 为什么不升反降，以及这个差距该怎么读。** 换成大模型后，CLIMATE-FEVER
上朴素聚合显著变差（约 −0.03），conflict_aware 也没有提升（−0.019，不显著）。
原因在它们共用的判定规则
（「统一判定规则」一节）：三个平均分里 `unknown` 最高就判 `insufficient`，这一步
没有可调的阈值。大模型给大多数文档很高的 Neutral 概率，平均之后 `unknown`
几乎总是最高 —— 616 条里 Weighted Average 把 563 条判成 `insufficient`；
conflict_aware 只判了 8 条 `supported`、37 条 `refuted`，其余 571 条都落在
`insufficient`（354）和 `conflicting`（217）。两维判定没有这个问题：证据够不够由
一个**单独的分数加可调阈值**决定，支持还是反驳只在两个方向之间比。所以这里的
+0.10 ~ +0.19 量出来的是「证据不足一维要单独设阈值、不能和支持反驳放在一起取
最大」的价值，是框架层面的结论；它与 Dempster 规则无关（`sr_std` 版 Statistic Gate
的横轴就是平均 `p_unknown` 加阈值，同样显著高于 conflict_aware）。conflict_aware 的
规则是在这组实验之前定下的，没有为大模型改过。

**嵌套消融**（`outputs/metrics/*cross_validation_deepseek_flash_ablation/`）：两个判定
维度各自显著有用，幅度比 NLI 输入时更大（CLIMATE-FEVER 上 NLI 输入时是
−0.081 / −0.054 / −0.139）。

| 变体 | CLIMATE-FEVER：Macro-F1 | Δ vs full | 95% CI | AVeriTeC：Macro-F1 | Δ vs full | 95% CI |
| --- | --- | --- | --- | --- | --- | --- |
| full | 0.5053 | — | — | 0.5781 | — | — |
| 去掉证据不足一维 | 0.3722 | −0.1331 | [−0.1578, −0.1066] | 0.4345 | −0.1436 | [−0.1684, −0.1185] |
| 去掉文档冲突一维 | 0.4191 | −0.0862 | [−0.1129, −0.0582] | 0.4575 | −0.1206 | [−0.1438, −0.0959] |
| 两维都去掉 | 0.2855 | −0.2197 | [−0.2560, −0.1812] | 0.3188 | −0.2593 | [−0.2910, −0.2257] |

D-S 有 16 条（CLIMATE-FEVER）和 5 条（AVeriTeC）因为支持与反驳的信度恰好相等而
落到 `undetermined`，按判错计；四个 Statistic Gate 各有 7–16 条。NLI 输入时一条
都没有 —— 口头概率取值粗，才会出现精确平局。

#### 检测：K_doc 在 AVeriTeC 上第一次超过「数篇数」，在 CLIMATE-FEVER 上仍不是最好

与「五」同一套分数、同一个脚本（`outputs/metrics/detection_proxies_deepseek_flash/`、
`outputs/metrics/averitec_detection_proxies_deepseek_flash/`）。冲突识别的 AUROC：

| 分数 | CLIMATE-FEVER | K_doc − 它 | p | AVeriTeC | K_doc − 它 | p |
| --- | --- | --- | --- | --- | --- | --- |
| **K_doc**（D-S） | 0.6522 | — | — | **0.7781** | — | — |
| `min_mean` | **0.7023** | −0.0501 | 0.016 | 0.7465 | +0.0317 | 0.099 |
| `min_max` | 0.6637 | −0.0115 | 0.582 | 0.7561 | +0.0220 | 0.144 |
| `max_product` | 0.6571 | −0.0049 | 0.598 | 0.7460 | +0.0321 | 0.022 |
| `mean_product` | 0.6342 | +0.0180 | 0.001 | 0.6911 | +0.0870 | < 0.001 |
| `sr_std` | 0.6253 | +0.0269 | 0.039 | 0.6986 | +0.0795 | < 0.001 |
| 只数证据篇数 | 0.5000 | +0.1522 | < 0.001 | 0.7141 | +0.0640 | < 0.001 |

NLI 输入时 K_doc 是 0.606 与 0.683。证据不足识别：`m_theta` 的 AUROC 是 0.786 与
0.815（NLI 输入时 0.621 与 0.645），与平均 / 最小 / 连乘 `p_unknown` 三个统计量
全部打平（差值都在 ±0.005 以内，均不显著）。

- AVeriTeC 上 NLI 输入时「只数篇数」（0.714）高于 K_doc（0.683），是「八」里最
  需要警惕的现象；换成大模型后 K_doc 升到 0.778，**显著高于**数篇数，也显著高于
  `max_product` 与 `sr_std`，与 `min_mean`、`min_max` 打平。篇数相同的 claim 内部，
  K_doc 的 AUROC 是 0.735（2 篇）、0.714（3 篇）、0.701（4 篇）、0.638（5 篇），
  NLI 输入时对应 0.584、0.623、0.552、0.579。
- CLIMATE-FEVER 上 `min_mean`（平均支持与平均反驳里较小的那个）显著高于 K_doc。
  「K_doc 作为检测分数不优于简单统计量」在这个数据集上仍然成立。

#### AVeriTeC 的证据篇数对照

「八」的教训是 AVeriTeC 上任何结论都要先过「证据篇数」这一关，这里原样重做：

| 设定 | D-S | `max_product` | conflict_aware（调充分） | D-S − conflict_aware | 95% CI | p |
| --- | --- | --- | --- | --- | --- | --- |
| 全部证据（920 条） | 0.5781 | 0.5987 | 0.4733 | +0.1047 | [+0.0693, +0.1400] | < 0.001 |
| 只取前 2 篇（763 条） | 0.4539 | 0.4593 | 0.3967 | +0.0572 | [+0.0145, +0.0996] | 0.007 |
| 只取前 3 篇（478 条） | 0.4823 | 0.4818 | 0.3625 | +0.1198 | [+0.0694, +0.1694] | < 0.001 |

NLI 输入时后两行是 +0.018（p = 0.37）与 +0.040（p = 0.109）。**控制篇数之后，D-S
对 conflict_aware 的优势仍然显著**，与 `max_product` 始终打平。`max_product` 两个
变体对 conflict_aware 在两种截断下也都显著；`sr_std` 两个变体在前 3 篇时显著，
前 2 篇时不显著（+0.020，p = 0.37）。

冲突一维只数篇数的对照（`stat_gate_doc_count`，另两维照常用大模型的关系概率）是
0.5667，D-S 比它高 0.0113（[−0.0202, +0.0433]，p = 0.47，**不显著**）。也就是说，
在全部证据上，冲突一维带来的四分类增益仍然有相当一部分可以用篇数解释；检测
AUROC 与前 K 篇截断才是 K_doc 确实读到了内容的证据。截断会丢证据，前 K 篇两行
是对照，不是 AVeriTeC 上的最好成绩。

#### 直接判四类与两维判定：同一个模型的两种用法

| 数据集 | 方法 | Macro-F1 | supported | refuted | insufficient | conflicting |
| --- | --- | --- | --- | --- | --- | --- |
| CLIMATE-FEVER | 直接判四类 | 0.4555 | 0.416 | 0.582 | 0.522 | 0.302 |
| CLIMATE-FEVER | D-S | 0.5053 | 0.567 | 0.559 | 0.534 | 0.361 |
| CLIMATE-FEVER | Statistic Gate（`max_product`） | 0.5276 | 0.561 | 0.597 | 0.571 | 0.381 |
| AVeriTeC | 直接判四类 | 0.5861 | 0.655 | 0.715 | 0.665 | 0.309 |
| AVeriTeC | D-S | 0.5781 | 0.587 | 0.662 | 0.608 | 0.455 |
| AVeriTeC | Statistic Gate（`max_product`） | 0.5987 | 0.638 | 0.677 | 0.609 | 0.470 |

（后四列是各类 F1。）配对检验见上面两张主表的「大模型直接判四类」一行，以及
`outputs/llm_judge/<数据集>/deepseek-flash/judge_summary_vs_deepseek_flash.json`：

- CLIMATE-FEVER 上，两维判定显著高于直接判四类：D-S +0.050（p = 0.020），
  `max_product` +0.072（p < 0.001）；两个 `sr_std` 变体与它打平；
- AVeriTeC 上三者**打平**（直接判四类比 D-S 高 0.008，比 `max_product` 低 0.013，
  都不显著）；
- 对 NLI 输入的全部方法，直接判四类都显著更高（+0.09 ~ +0.33，
  `judge_summary.json`）—— 一个零样本的大模型，胜过 NLI 小模型加任何聚合方式；
- 直接判四类在两个数据集上都**很少判冲突**：`conflicting` 的召回率是 0.195 与
  0.196（精确率 0.67 与 0.74），F1 只有 0.30 左右；两维判定的冲突召回率是
  0.27–0.36，F1 0.36–0.47。AVeriTeC 上直接判四类在另外三类更强，两者相抵。

这组比较**不能**读成「两维判定优于大模型直接判」。三个必须写明的不对等：

1. 两维判定的阈值在带标签的数据上选（嵌套交叉验证的内层），直接判四类是零样本、
   完全不看标签。给直接判四类加几条示例，结果很可能不同，这里没有做。
2. CLIMATE-FEVER 的 claim 标签本来就是「逐篇证据投票再按规则合成」的，逐篇判断再
   聚合的方法在结构上贴合造标签的方式（「八」开头说过）。标签由人直接判定的
   AVeriTeC 上，两者打平 —— 后者才是更可信的比较。
3. 两维判定每条 claim 要发「文档数」个请求，直接判四类只发一个。

能说的是：**在同一个大模型上，逐篇判断加两维判定不比一次性直接判差，并且在
「文档互相矛盾」这一类上明显更敏感**，同时给出可以检查的中间量（每篇文档的三个
概率、证据不足与冲突两个分数）。

#### 对前面结论的更新，与这组实验的限制

| 前面的结论（NLI 输入） | 换成大模型后 |
| --- | --- |
| 瓶颈在关系评估器（「四」，只有人工上限作证据） | **证实**：只换评估器，D-S +0.15 / +0.18 |
| 两维判定对调充分的 conflict_aware 不显著（「四」「八」） | **改变**：显著，+0.10 ~ +0.19，5/5 个种子；AVeriTeC 控制篇数后仍显著 |
| D-S 与四个 Statistic Gate 打平（「四」） | **部分改变**：仍不优于 `max_product`（点估计略低）；显著优于 `sr_std` |
| 两个判定维度各自显著有用（「六」） | **证实**，幅度更大 |
| K_doc / m_theta 作检测分数不优于简单统计量（「五」） | CLIMATE-FEVER 上不变；AVeriTeC 上 K_doc 最高，但与 `min_mean`、`min_max` 不显著 |
| AVeriTeC 上「只数篇数」胜过 K_doc（「八」） | **改变**：K_doc 的 AUROC 显著更高；四分类上数篇数的对照仍与 D-S 打平 |

限制：

- **一个模型、一份提示词、各跑一遍。** 采样温度用的是接口默认值，重跑不会逐条
  相同；没有做提示词敏感性，也没有第二个大模型。`deepseek-v4-pro` 的判四类因
  账户余额中断，只完成了 556 / 616 与 676 / 920 条（断点文件在
  `outputs/llm_judge/<数据集>/deepseek-v4-pro/`，没有汇总），**不完整，不能引用**。
- 口头概率取值粗，未做校准。
- 多评估器（NLI + 大模型）、估计可靠性、选择性回答没有在大模型输入上重做；
  CONFACT 也没有重跑。
- 大模型可能在训练时见过这两个公开数据集。提示词要求只依据给出的文档，但无法
  排除记忆的影响；直接判四类与关系评估器同样受此影响。

## 目录说明

- `configs/`：项目配置文件。
- `data/raw/`：未经处理的原始数据。
- `data/processed/`：清洗或转换后的数据。
- `data/samples/`：可提交到仓库的小型样例数据。
- `src/rag_ds/`：项目的 Python 源代码包。
- `scripts/`：后续用于运行数据处理或实验的命令行脚本。
- `tests/`：自动化测试。
- `outputs/predictions/`：模型或评估器的预测输出。
- `outputs/ragchecker/`：RAGChecker 的输入/输出、模型关系文件与谱系清单。
- `outputs/llm_judge/`：大模型直接判四类的逐条结果与汇总。
- `outputs/metrics/`：评估指标输出。
- `outputs/figures/`：图表输出。

