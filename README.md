# rag-ds-evaluator

用于研究 claim-level RAG 证据状态诊断的 Python 项目。

当前进度：总规划的 14 个生成步骤已完成 1–14。
适配器层（RAGChecker / RAGAS / LLM）只有**接口与转换**，不导入也不调用
这些第三方库；SURE-RAG 尚未接入。

第一版正式数据 `data/processed/climate_fever_v1/` 已从 CLIMATE-FEVER 构建完成
（400 条、四类各 100、train/validation/test = 240/80/80）。关系文件是人工证据
投票的 **annotation oracle**，只用于验证 D-S 融合链路，**不能**当作模型实验
结果写入论文。多评估器真实实验、RAGChecker / RAGAS 真实输出接入均尚未开展。

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

### 三种方法

| 方法 | 规则 | 用到的可靠性 |
| --- | --- | --- |
| **Weighted Average** | 对全部评估器 × 全部文档的三个概率做加权平均 | 文档可靠性 × 评估器可靠性 |
| **Majority Vote** | 每条关系预测一票，少数服从多数 | **都不用**（一票一权） |
| **Single Evaluator** | 只用一个指定评估器，按文档可靠性加权平均 | 只用文档可靠性 |

Majority Vote 中单条预测内部出现并列最大值时投 `unknown` 票 —— 该条本身就
分不清方向，不替它选边。

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

### 三个 baseline 都不会输出 conflicting

这是**刻意的设计**，也是实验要展示的核心局限：朴素聚合把「两条针锋相对的
证据」压成一个低分或一个平局，无法与「谁都说不清楚」区分开。
`BaselinePrediction` 在模型层就禁止 `predicted_state = conflicting`。

D-S 方法用 `K_doc` 把这件事显式量化出来，因此能给出 `conflicting`。

### demo 上的实际对照

| claim | 标注 | D-S | Weighted Average | Majority Vote | Single Evaluator |
| --- | --- | --- | --- | --- | --- |
| demo-001-c1 | supported | **supported** | insufficient | insufficient | insufficient |
| demo-001-c2 | supported | **supported** | supported | supported | supported |
| demo-002-c1 | refuted | **refuted** | refuted | refuted | refuted |
| demo-003-c1 | insufficient | **insufficient** | insufficient | insufficient | insufficient |
| demo-004-c1 | conflicting | **conflicting** | insufficient | insufficient | insufficient |

两处 baseline 失手：

- **demo-004-c1**（标注 conflicting）—— 一条支持、一条反驳的文档被压成
  0.463 / 0.487 / 0.05，最高分不到 0.5，判为 `below_threshold`；投票则是
  1:1 的 `score_tie`。三个 baseline 都只能说「不确定」，说不出「有冲突」。
- **demo-001-c1**（标注 supported）—— 一条强支持文档（0.90）与一条无信息
  文档（unknown 0.90）平均后得到 0.486 / 0.050 / 0.464，同样卡在阈值下方。
  D-S 的 Dempster 组合让无信息证据自然让位，得到 m(S) = 0.856。

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

## 目录说明

- `configs/`：项目配置文件。
- `data/raw/`：未经处理的原始数据。
- `data/processed/`：清洗或转换后的数据。
- `data/samples/`：可提交到仓库的小型样例数据。
- `src/rag_ds/`：项目的 Python 源代码包。
- `scripts/`：后续用于运行数据处理或实验的命令行脚本。
- `tests/`：自动化测试。
- `outputs/predictions/`：模型或评估器的预测输出。
- `outputs/metrics/`：评估指标输出。
- `outputs/figures/`：图表输出。

