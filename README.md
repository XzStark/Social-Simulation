# pepole — 政策与产品发行动作的社会 / 市场反响演练

**许可证**：[MIT](./LICENSE) · **参与贡献**：[CONTRIBUTING.md](./CONTRIBUTING.md) · **安全漏洞上报**：[SECURITY.md](./SECURITY.md)  
**维护者（网名 STARK）GitHub 主页**：[github.com/XzStark](https://github.com/XzStark)

**怎么操作**：请看根目录 **[操作指南.md](./操作指南.md)**（安装、命令、改 YAML、指标含义、示例场景）。  
**系统说明（架构与机制）**：[系统说明.md](./系统说明.md)（模块职责、数据流、API、概念词典，偏技术/细致）。  
**扩展能力、验证校准、决策辅助（曲线粘性 + 粗检）、蒙特卡洛、缺口与路线图**：见 **[系统说明.md](./系统说明.md)**（**§6.1**、§8.1、§12～§19）。  
**政府 / 企业 / 创业团队** 的责任边界、数据与模型治理：**[系统说明.md](./系统说明.md) §1.4～§1.5**。  
**网页界面**：先在本机设置 `OPENAI_API_KEY`（及可选 `OPENAI_BASE_URL`），再运行 **`run_web.cmd`**（默认端口 **8770**）或 `python -m uvicorn pepole.webapp:app --host 127.0.0.1 --port 8770` → http://127.0.0.1:8770  
若报端口占用，运行 **`结束端口占用.cmd`** 或改 `run_web.cmd` 里的 `PORT`。  
**模型与密钥**：OpenAI / ChatGPT API → **[配置示例-OpenAI-ChatGPT.cmd](./配置示例-OpenAI-ChatGPT.cmd)**；Gemini → **[配置示例-Gemini.cmd](./配置示例-Gemini.cmd)**；Claude → **[配置示例-Claude.cmd](./配置示例-Claude.cmd)**；DeepSeek → **[配置示例-DeepSeek.cmd](./配置示例-DeepSeek.cmd)**。将占位符改为你的密钥后，在该 CMD 窗口 `call` 对应文件再启动。**勿将含真实密钥的文件提交到 Git**（可用 `*.local.cmd`，见 `.gitignore`）。汇总见 **[.env.example](./.env.example)**。

面向 **中央 / 省 / 市 / 县**各级政策制定者与 **大型集团、中小企业、创业公司**高层：在沙盘里**下达**政策或产品方案（`player_brief`），观察舆情、支持度、动荡、**主体信任**与**供应链压力**等；可粘贴**脱敏真实案例**（`reference_cases_brief` / `--reference-file`）供模型对照机制。海量公众用 **cohort** 聚合，并按 **基层 / 中产 / 高层 proxy**（`class_layer`）区分人性与动员逻辑；具名角色用人设名册，规模可很大而不等于每轮同等次数 LLM。

**定位说明**：输出是「压力测试与叙事推演」，不能替代统计调查与法务结论；系统与提示词刻意**去除游戏感**（禁用通关、Boss、buff 等隐喻），用语贴近机关与企业内参语境。  
**机构级使用**：涉及涉密与个人信息、对外引用演练结论、采购与等保关基对齐时，请先阅读 **系统说明 §1.5、§19**（本项目**不宣称**已通过特定强制性测评；评估与加固由部署方负责）。

## 两种演练类型

| `exercise_type` | 典型用户 | 你输入什么 | 指标怎么读 |
|-----------------|----------|------------|------------|
| **`policy`** | 政策制定者、监管部门 | 法规/政策要点、过渡期、执法口径等 | `policy_support` = 对政策/施政的接受度；`unrest` = 动荡/对抗强度 |
| **`product`** | 集团发行方（**软件 / 硬件 / 软硬一体**均可） | 产品形态、定价、渠道、合规与传播、`player_brief` 写清即可 | `policy_support` = **对发行方动作的支持/好感（含购买意愿 proxy）**；`unrest` = 争议与抵制强度 proxy；`issuer_trust_proxy` = 品牌/主体社会信任慢变量；`supply_chain_stress` = 交付与外部冲击 proxy |

- **行政层级**：`policy_context.admin_level` = `central` / `province` / `city` / `county`，配合 `jurisdiction_name`、`local_norms_brief`、`media_environment_brief`（民风、人情、媒体结构），模型不得「全国一张脸」。示例：`scenarios/policy_county_example.yaml`（县）、`scenarios/default.yaml`（省）。
- **集团类型与品牌**：`issuer.archetype` = `megacorp` / `large_group` / `sme` / `startup`，`brand_equity`（0～1）与 `reputation_brief`、`supply_chain_position` 进入 `decision_context`；**同一舆情**对国民品牌、巨头与冷启动品牌的**边际伤害不同**（由提示词约束 + 规则慢变量体现，而非桌游数值）。
- **阶层**：每个 cohort 可选 **`class_layer`**：`lower` / `middle` / `upper` / `mixed`，影响舆情加权与动荡贡献（基层更易被生计与公平感动员——见 `realism` 中阶层权重）。
- **软件 / 硬件**：仍由 **`player_brief` + `simulation.product_kind` + 人设 `product_kinds` / `categories`（含 `supply_chain`）** 表达。

---

## 「红楼梦式」80+ 智能体怎么落地？

- **人设（persona）**：在 YAML 的 `personas` 里一人一条：`id`、`role`、`goals`、长文 **`persona`**（小传）、**`categories`**（购买者/竞品/律师/境外监管等）。
- **不等于每 tick 调 80 次模型**：名册可以 **80+**，每轮只 **唤醒** 一部分：
  - **`llm_each_tick: true`**：每 tick 必调用（如核心监管叙事、主编）。
  - **池化人设**：由 **`simulation.pooled_llm_calls_per_tick`** 控制每 tick 从池里 **加权无放回** 抽几个；**`always_sample_ids`** 可固定「首席竞品」「欧盟合规顾问」等优先占坑。
- **世界范围**：**`simulation.markets_active`**（`domestic` / `export`）与 **`product_kind`**（`software` / `hardware` / `hybrid` / `general`）会过滤人设：只让「会登场」的角色进池（例如只做国内硬件的渠道商不会在你勾选「仅软件+出口」时进池）。
- **大众面**：仍用 **`cohorts`** 表示千万级购买者/公众；需要时再开 **`cohort_llm`** 用轻量模型批量推态度。

这样可满足：**购买者是否买账、竞品、打官司叙事、出口与境外规则** 等角色都存在，且成本可控。

---

## 快速开始

```bash
cd D:\pepole
pip install -r requirements.txt
python main.py run --scenario scenarios/default.yaml
python main.py run --scenario scenarios/product_launch.yaml
# 多角色 + 国内/出口 + 竞品/诉讼/买家（人设示例，可扩到 80+）
python main.py run --scenario scenarios/society_export_software.yaml --show-budget
python main.py run --scenario scenarios/default.yaml --brief "单行政策要点……"
python main.py run --scenario scenarios/policy_county_example.yaml
python main.py run --reference-file ./cases/redacted_case.txt
python main.py ensemble --scenario scenarios/product_launch.yaml --runs 8 --workers 4 --dump-json out.json
# 扩展栈演示（因果/锚定/KPI/扩散/时延/资源池等，见 系统说明.md §12）
python main.py run --scenario scenarios/roadmap_demo.yaml --dump-metrics-json metrics.json
python main.py validate --truth-csv scenarios/roadmap_demo_truth_sample.csv --sim-json metrics.json
# 校准闭环（对照真值自动试 realism 参数）与敏感性、稳定性、通俗归因解读 — 见 系统说明.md §13
python main.py calibrate --scenario scenarios/roadmap_demo.yaml --truth-csv scenarios/roadmap_demo_truth_sample.csv --grid "llm_effect_multiplier:0.28,0.38,0.48" --seed 42
python main.py stability --scenario scenarios/product_launch.yaml --runs 12
python main.py run --scenario scenarios/default.yaml --dump-full-state run_dump.json
python main.py explain --from-json run_dump.json
# 网页端并行分布：POST /api/ensemble（与 RunRequest 同字段 + runs / workers / threshold_*）
```

### 环境变量与多厂商密钥

| 变量 | 用途 |
|------|------|
| `OPENAI_API_KEY` | **OpenAI 官方 API**（网页 ChatGPT 与 API 密钥不同）或 **兼容网关**（DeepSeek、Azure、Together 等，配合 `OPENAI_BASE_URL`） |
| `OPENAI_BASE_URL` | 可选；默认 `https://api.openai.com/v1` |
| `ANTHROPIC_API_KEY` | **Claude**（Anthropic 控制台申请） |
| `GOOGLE_API_KEY` 或 `GEMINI_API_KEY` | **Gemini**（Google AI Studio 申请，两个变量名任选其一） |
| `PEPOLE_MODEL_PRIMARY` / `PEPOLE_MODEL_FAST` | 路由串，格式 **`厂商前缀:模型名`**（见下表） |

**路由前缀（`main.py` / 网页「模型」框同理）**：

| 前缀 | 厂商 | 示例模型名 |
|------|------|------------|
| `openai:` | OpenAI 兼容 HTTP（含官方与 DeepSeek 等） | `gpt-4o`、`gpt-4o-mini`、`deepseek-reasoner` |
| `anthropic:` | Claude | `claude-sonnet-4-20250514` |
| `google:` 或 `gemini:` | Gemini | `gemini-2.0-flash` |

Windows 下一键写入环境变量：根目录 **`配置示例-OpenAI-ChatGPT.cmd`**、**`配置示例-Gemini.cmd`**、**`配置示例-Claude.cmd`**、**`配置示例-DeepSeek.cmd`**（把占位符改成你的密钥后，在 CMD 里 `call` 该文件再运行程序）。通用模板见 **[.env.example](./.env.example)**。

无对应厂商 Key 时为 **Dry-Run**（可走通流程，输出占位）。

---

## 场景 YAML 要点

| 字段 | 作用 |
|------|------|
| `player_brief` / `exercise_type` | 用户动作与沙盘类型 |
| `policy_context` | 政策侧：**哪一级**、何地、民风与媒体环境 |
| `issuer` | 产品侧：**集团体量**、品牌资产、声誉叙事、供应链位置 |
| `reference_cases_brief` | 脱敏真实案例摘要；CLI 可用 `--reference-file` |
| `cohorts[].class_layer` | 基层 / 中产 / 高层 proxy，驱动聚合权重与人性差异 |
| `key_actors` | 兼容旧版：**每人每 tick 必调 LLM**（可填 `persona`） |
| `personas` | 人设名册；配合 `llm_each_tick` / 池化抽样 |
| `simulation` | `product_kind`、`markets_active`、`pooled_llm_calls_per_tick`、`always_sample_ids` |
| `triggers` | **条件触发**：当指标满足 AND 条件时，将指定人设并入本 tick 的「优先占池化名额」（如高舆情 → 律师团） |
| `realism` | **真实感参数**：环境噪声、社会/支持度惯性、对 LLM 输出 delta 的再缩放与硬顶（减轻「游戏式」单轮翻盘） |
| `cohorts` | 海量人群桶 |
| `cohort_llm` | 是否为各桶加轻量 LLM |

完整示例见 `scenarios/society_export_software.yaml`（含 `triggers`、`realism`、财务/供应链/黑产钓鱼等扩展人设）。

### 为何不会像游戏？

- **规则层**：舆情、支持度、**主体信任**、**供应链压力**有**惯性、衰减与耦合**，LLM 输出再经**缩放与封顶**；阶层加权使「底层动静」更贴近现实动员结构。  
- **叙事层**：提示词明确**受众为决策者**、**禁用游戏隐喻**，并要求区分**行政层级与品牌资产**，尊重程序与时滞。  
- **编排层**：人设再多，每 tick 仍只激活**固定 + 抽样 + 条件触发**的子集。

---

## API 与调用量

设 \(T\)=`ticks`，\(N\)=`--runs`，cohort 批量关闭时：

- 令 \(A_{\min}, A_{\max}\) = 每 tick LLM 智能体数下界/上界（`python main.py run --show-budget` 可打印）。
- **PRIMARY / run** \(\approx T \times A\)（\(A\) 在 \([A_{\min}, A_{\max}]\) 之间，池化每 tick 可能不同但期望由配置决定）。

旧式仅 `key_actors`、无 `personas` 时：\(A = |key\_actors|\)。

若 `cohort_llm.enabled: true`：每 tick 约 **+1** 次 FAST（batch 全 cohort）。

---

## 克隆与开发

需要 **Python 3.10+**。克隆后：

```bash
cd pepole
pip install -r requirements.txt
# 可选：可编辑安装 + 测试依赖
pip install -e ".[dev]"
python -m pytest
```

命令行入口仍为项目根目录的 **`main.py`**（例如 `python main.py run --scenario scenarios/default.yaml`）；网页见上文 **`run_web.cmd`**。参与贡献与安全上报见 **[CONTRIBUTING.md](./CONTRIBUTING.md)**、**[SECURITY.md](./SECURITY.md)**。

---

## 许可

以根目录 [LICENSE](./LICENSE) 为准（MIT）。著作权人以网名 **STARK** 署名（个人维护项目）。
