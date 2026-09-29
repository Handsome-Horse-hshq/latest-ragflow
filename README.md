# rag-ds-evaluator

用于研究 claim-level RAG 证据状态诊断的 Python 项目。

当前进度：总规划的 14 个生成步骤已完成 1–14，**RefChecker NLI checker 的
真实输出已接入并在 v1 / v2 两个数据集上跑出结果**（导出输入 → 推理 →
摄取 → 谱系记录 → 阈值搜索 → 显著性检验 → 嵌套交叉验证）。
适配器层（RAGChecker 全流程 / RAGAS / LLM）仍然只有**接口与转换**，不导入也
不调用这些第三方库；RAGAS 接入与 SURE-RAG 尚未开展。多评估器实验已在 v2 上
完成（RefChecker NLI 管线 × 两个 NLI 模型），K_eval 有真实取值。

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
| Conflict Aware | 0.3523 | 0.3074 | [0.2738, 0.3419] |
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
输出空间，一半来自融合机制，两层各自都仍显著。三点必须如实报告：

1. **加入第二个评估器后 D-S 的 Macro-F1 从 0.3535 降到 0.3320** —— 多评估器
   融合没有带来增益。两个 NLI 模型同源同任务，错误高度相关，第二个评估器
   更多是稀释了第一个的有效信号而不是补充新信息。
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

## 三处必须正视的问题及其处理

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

### 二、加第二个评估器反而变差

单评估器 Macro-F1 0.3535 → 双评估器 0.3320（估计可靠性可回到 0.3466，仍更低）。
这与"多源融合更好"的直觉相反，原因是**两个评估器的误差高度相关**：

| 指标 | 取值 |
| --- | --- |
| 三个概率通道的 Pearson r | +0.71 ~ +0.74 |
| argmax 标签一致率 | 85.1%（相互独立时期望仅 60.6%） |
| Cohen's kappa | **0.623** |

**Dempster 组合规则要求证据源相互独立。** 两个同源 NLI 模型
（都出自 MNLI/FEVER/ANLI 家族）在重复计算同一份证据，质量被虚假锐化，
融合因此不增反减。双评估器下 K_eval 告警只占 3.7% 也是同一现象的表现。

这正是引入机制异构的第三个评估器（生成式 flan-t5，见
`scripts/run_t5_judge_chunked.py`）的动机。

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

修正后 D-S 在两个指标上都领先（AUROC 0.576 → 0.621，baseline 全在 0.51~0.54）。
但**并非全部翻转**：`max_signal` 下 AUROC 仍是 `weighted_average` 略高，
两套数字论文里都要给出。

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
- `outputs/metrics/`：评估指标输出。
- `outputs/figures/`：图表输出。

