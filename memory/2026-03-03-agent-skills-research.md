# Session: 2026-03-03 17:15:37 UTC

- **Session Key**: agent:meta-research:telegram:direct:8395048267
- **Session ID**: 48e1bb7a-cd27-4ac6-bf56-066c1f47b467
- **Source**: telegram

## Conversation Summary

assistant: Sub-agent is running. While it digs deep, I'm doing parallel targeted fetches across the wild-card disciplines you won't find just by searching arXiv:
assistant: More gold — SAGE (RL for skill libraries, Dec 2025), MCP quality "smells" paper, AdapTools (indirect prompt injection via MCP). Let me fetch those abstracts and simultaneously write everything into a comprehensive brainstorm document:
assistant: Now I have enough to synthesize the full brainstorm. Writing the comprehensive document now:
assistant: Sub-agent is still running its parallel deep search. Here's what the expanded picture looks like:

---

## 🌍 What the Deep Dive Revealed

### 🔥 The Big Shift
The field is **4 months old**. Anthropic launched Agent Skills in October 2025 → 62,000 ⭐ in 4 months. The paradigm evolution is:
```
Prompt Engineering (2022-23)  →  Tool Use (2023-24)  →  Skill Engineering (2025+)
    ephemeral, non-modular         atomic, execute-only     composable, procedural, community-governed
```
**We are writing about the third paradigm as it's being born.** First-mover window: ~6–12 months.

---

### 🗺️ 10 New Research Clusters (beyond original 5 hypotheses)

| Cluster | Headline Opportunity | Why Urgent |
|---------|---------------------|-----------|
| **Quality Framework** | *SkillSmell* — measure SKILL.md quality → performance | MCP smells paper proved 11.6% effect. Nobody did this for full skills yet. **No GPU needed.** |
| **Security** | *SkillGuard* — 26.1% vulnerable, zero defense papers | Growing attack surface, enterprise adoption, IEEE S&P / ICLR Safety |
| **Architecture** | Progressive disclosure effectiveness, skills vs. tools vs. RAG | Clean ablation, theoretical grounding |
| **Cross-backbone** | Portability gaps + Universal Skill spec | Explicitly flagged open by Feb 2026 survey |
| **Ecosystem** | Community dynamics, economics, niche formation | Virgin territory — no AI skill ecosystem studies exist |
| **Cognitive Science** | Skills as externalized procedural memory (CLS theory) | Interdisciplinary reach — CogSci, NeurIPS cognitive track |
| **Automation Loop** | Formalizing the session feedback → skill improvement cycle | Platform's core mechanism, never formally studied |
| **Temporal** | Skill decay and freshness as frameworks evolve | Practical, underexplored, EMNLP angle |
| **Multimodal** | Visual skills for GUI agents (OSWorld/WebArena) | GUI agents are the deployment frontier |
| **Wild Cards** | Skill grounding via executable tests, version control for knowledge | Software engineering meets AI cognition |

### 🎯 Revised Priority Stack

1. **SkillSmell** (NeurIPS 2026) — quality framework, starts today, no GPU
2. **SkillGuard** (ICLR 2027 Safety) — security, urgent, no competition
3. **Portability** (ICML 2026) — explicitly open problem, clean experiment  
4. **Craft vs. Canon** (NeurIPS 2026) — community knowledge value, needs Orchestra data

Sub-agent is still mining — will announce when done. Want to dive into any specific cluster deeper right now?
assistant: ✅ Subagent meta-research finished

**Done.** Here's what was accomplished:

## Summary

**Deep Cross-Disciplinary Literature Survey** written and committed to `skill-evolve-research/literature/deep-survey.md`.

### What was produced (~50KB, 694 lines):

**PRISMA numbers:** 138 papers found → 99 screened → 68 included across 10 clusters

**Per-cluster findings with full extractions** for:
- **Cluster A (Core AI/ML):** 18 papers — Voyager, SkillRL, ProcMEM, AutoRefine, AWM, MemSkill, ToolLLM, GentPool, Agent Skills Survey (2602.12430), and more
- **Cluster B (Collective Intelligence):** Kittur/Wikipedia, human-LLM hybrid annotation, crowdsourcing quality
- **Cluster C (Knowledge Management):** SECI model (Nonaka), CodeMem, Evolutionary Context Search
- **Cluster D (DevEx/SE):** ToolBench, OpenAgents, AgentBench, JARVIS-1
- **Cluster E (OSS Dynamics):** npm/PyPI ecosystem dynamics, plugin marketplaces
- **Cluster F (Security):** 9 papers — MCPShield, SMCP, AdapTools, MemoryGraft, misleading MCP descriptions
- **Clusters G-J:** Benchmarks, cognitive science/CLS theory, incentive design, MCP protocol evolution

**Key deliverables inside the document:**
1. PRISMA-style inclusion table
2. Full paper extractions (citation, claim, method, result, relevance, gap)
3. 6 cross-cutting themes (Procedural Memory Gap, Quality-Autonomy Tension, Security as First-Class, Evaluation Problem, Protocol Convergence, Evolutionary Dynamics)
4. 6 surprising finds (26.1% vulnerability rate, procedural memory research explosion in late 2025, zero papers on multi-agent skill evolution)
5. **10 white spaces** with zero prior coverage — including cross-agent portability, meta-skill quality, skill lifecycle models
6. **Top 20 papers** with full extraction schema
7. **10 new research opportunities** ranked by novelty × impact
8. **Evidence map** linking all papers to H1-H5

**Most important finding:** There is **zero prior academic work** at the intersection of Skill Architecture ∩ Collective Intelligence ∩ Security Governance — which is exactly where SkillEvolve operates. The field is wide open.
user: Conversation info (untrusted metadata):
```json
{
  "timestamp": "Tue 2026-03-03 16:19 UTC"
}
```

Make sure to follow the meta research instructions [https://clawhub.ai/AmberLJC/meta-research](https://clawhub.ai/AmberLJC/meta-research)
user: Conversation info (untrusted metadata):
```json
{
  "timestamp": "Tue 2026-03-03 16:21 UTC"
}
```

Make sure to follow the meta research instructions [https://clawhub.ai/AmberLJC/meta-research](https://clawhub.ai/AmberLJC/meta-research)
assistant: On it. Let me load the full phase instructions and run the proper workflow from where we are.
assistant: Now let me also load the ideation frameworks and judgment rubric before proceeding:
user: Conversation info (untrusted metadata):
```json
{
  "timestamp": "Tue 2026-03-03 17:04 UTC"
}
```

怎么样
assistant: 正在推进！我已经读完了所有 meta-research 流程文件。现在按规范走完整个流程：更新 research-tree、运行 Hypothesis Generation（含 ideation frameworks + 两句话测试）、然后 Judgment Gate 全部假设打分。

**Phase: Lit Survey → Hypothesis Generation → Judgment Gate**
assistant: Now write the judgment summary and log, then commit:
assistant: ✅ 完成。以下是当前进展全景：

---

## 📍 当前状态

```
Literature Survey ✅ → Hypothesis Generation ✅ → Judgment Gate ✅ → Experiment Design (进行中)
```

---

## ⚖️ Judgment Gate 结果（共 14 个假设）

| 结果 | 假设 | 重要性 |
|------|------|--------|
| ✅ **APPROVED × 11** | H1, H1.1, H1.2, H2, H2.1, H3, H4, H5, H6, H7, H9 | 全部 HIGH（H1.2 MEDIUM） |
| ⏸ **DEFERRED × 2** | H8（需先有 H1+H6）, H10（需要 Orchestra 平台数据） | 依赖条件未满足 |
| ⏳ **PENDING × 1** | H3.1（条件性，等 H3 结果后决定是否启动） | — |
| ❌ **REJECTED** | 0 | — |

---

## 🌊 实验执行顺序（3 波次）

**Wave 1（现在就能开始，不需要 GPU）：**
- **H6 SkillSmell** — 构建 SKILL.md 质量框架，是所有其他实验的基础工具
- **H5 生态系统** — 纯 git log 数据分析，与 H6 并行
- **H7 安全漏洞** — 对 85 个 skills 做 red-team 分析

**Wave 2（需要 SkillBench 基础设施）：** H1, H2, H3, H9

**Wave 3（依赖 Wave 2）：** H4, H1.1, H1.2, H2.1

---

**要立刻开始 Wave 1 的实验设计吗？** 建议从 H6（SkillSmell）入手——不需要 GPU、直接产出成果、还能解锁后续所有假设。
user: Conversation info (untrusted metadata):
```json
{
  "timestamp": "Tue 2026-03-03 17:14 UTC"
}
```

全部停止所有任务
assistant: 全部停止了：

- ✅ **Cron job** — 已不存在（上次 gateway 重启时自动清除了）
- ✅ **Sub-agents** — 无活跃进程
- ✅ **后台任务** — 无

所有任务已停止。研究进度已保存在 GitHub：[AmberLJC/skill-evolve-research](https://github.com/AmberLJC/skill-evolve-research)，随时可以继续。
