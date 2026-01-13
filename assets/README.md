# Assets checklist

把论文 PDF 里的对应图片导出为 PNG（建议 2×/3×，宽度 ≥ 1600px），放到本目录并按下面文件名命名即可自动出现在页面中。

论文链接：`https://arxiv.org/pdf/2508.18983`

下面是**文件名 → 论文图号（Fig. X）→ 语义（你在论文里看到的那张图）** 的精确对照（按论文 caption 匹配）：

- `fig1.png` → **Fig. 1**：传统 MoE 层 vs 我们的 substitution 调度器（用低分专家替换 + 预取高分专家的总体框架图）
- `fig-pipeline.png` → **Fig. 9**：SMoE 在相邻两层 MoE 之间的流水线（GPU/CPU/Load 的并行与交叠示意）
- `fig-latency.png` → **Fig. 11**：TPOT（解码每 token 时间）——四个 baseline 与我们方法在 5 个 workload 上的对比（网页里“Latency”主结果）
- `fig-hit-rate.png` → **Fig. 12**：GPU cache ratio（平均缓存命中/驻留比例，网页里“Cache behavior”主结果）
- `fig-ablation.png`（可选但强烈推荐）→ **Fig. 14**：组件影响（Impact of components on TPOT and cache ratio，网页里“消融/分解”位）

想让网页更“高级感”但不依赖外部素材的话，建议再多放 1～3 张“解释机制/瓶颈”的图（可选）：



- `fig-bottleneck.png` → **Fig. 3**：CPU/GPU（A6000）计算时间 + PCIe 加载时间（3 个 MoE LLM），适合做“瓶颈来源”论证
- `fig-pcie-vs-tpot.png` → **Fig. 16**：PCIe time vs TPOT，适合做“为什么 TPOT 能下降”的解释
- `fig-components-load.png` → **Fig. 15**：需要加载的 top/low-score 专家数量更少，适合补充“替换/路由带来的 load 收敛”

如果你要把网页的 Key Findings 做得更像 VideoNSA 那种“问答式/逐条论证”，建议再导出这 3 张（可选但很推荐）：

- `fig-cpu-vs-gpu-lowbatch.png` → **Fig. 8**：GPU（A6000）vs CPU（8-core）在 low-batch 下的对比（强调 CPU 不是理想 fallback）
- `fig-prefill-time.png` → **Fig. 13**：Prefilling time（baseline vs SMoE 平均），补充端到端延迟叙事
- `fig-prefetch-accuracy.png` → **Fig. 17**：Prefetching accuracy（说明预取策略对质量的影响可控）




导出小建议（省事且观感更好）：
- 优先从 PDF 导出为 PNG（避免截图锯齿），背景尽量保持白底；网页会自动做暗色背景承托。
- 如果图太“扁”，可以在导出时加一点边距（留白），网页里的 contain 显示会更好看。


