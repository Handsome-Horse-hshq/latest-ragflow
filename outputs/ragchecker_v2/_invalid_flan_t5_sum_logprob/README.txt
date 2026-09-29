这些是 flan-t5 评判器早期版本的输出，已作废，不得用于任何实验。

原因（2026-09-29 查明）：
1. 打分用候选词各 token 对数概率之和。"entailment" 在 flan-t5 分词器下是
   4 个 token，"neutral"/"contradiction" 各 1 个，求和使 entailment 结构上
   几乎不可能胜出。
2. 提示 "nli premise: ... hypothesis: ..." 不是 FLAN 的原生模板，模型默认答 neutral。

实测 validation 620 条：99.8% 判为 unknown，0% 判为 support，
与人工 oracle 的 Cohen's kappa = +0.002（随机水平）。

修正版见 scripts/run_t5_judge_chunked.py（FLAN 原生 MNLI 模板 + 选项首 token 打分），
评估器名改为 flan_t5_mnli，以免与本目录的旧结果混淆。
