# SAG2 风格本地 Demo（教学用）

用最少代码演示 **SAG2 查询时多跳检索** 的思路：原文进 SQLite FTS5 → BM25/FTS 召回 →（演示用）证据打分 → 从有用段落抽词再搜。

## 这是什么 / 不是什么

| | |
|---|---|
| **是** | 可本地立刻跑的教育 Demo，展示「passage → term → passage」多跳 |
| **不是** | 论文复现；**证据判断不是真正的 JEV**（TypeSafe），只是关键词/实体启发式 |
| **不需要** | Embedding、向量库、GPU、API Key（默认路径） |

真实 SAG2：建库不调模型；检索时用 **BM25 + JEV** 做必要证据判断，再扩展词项。详见 [SAG-Benchmark](https://github.com/Zleap-AI/SAG-Benchmark)、[SAG Wiki](https://github.com/Zleap-AI/SAG-Wiki)、[技术说明](https://sag-wiki.zleap.com/en/docs/sag2)。

## 怎么跑

```bash
cd /workspace/sag2-demo
python3 sag2_demo.py
# 或自定义问题：
python3 sag2_demo.py "张三后来去了哪个项目？"
```

无需 `pip install`（纯标准库 `sqlite3`）。若你习惯用虚拟环境：

```bash
python3 -m venv .venv && source .venv/bin/activate
# requirements.txt 为空依赖，可跳过 pip install
python sag2_demo.py
```

默认问题：

> 收购了B公司的企业，其CTO后来加入了哪个项目？

期望证据链：`B公司/收购` → `A公司` → `张三/CTO` → `盘古大模型项目`（对应 `data/docs/01`–`03`）。

## 和真实 SAG2 的对应

| 步骤 | 本 Demo | 真实 SAG2 |
|---|---|---|
| 建库 | SQLite FTS5 + **汉字 bigram**（无依赖） | SQLite FTS5，原文词项索引 |
| 召回 | FTS5 `bm25()` | BM25 |
| 证据判断 | **启发式打分（标成 DEMO）** | **JEV**（需服务/密钥） |
| 扩展 | 从保留段落抽实体再搜 1–2 轮 | 从高分证据抽词，默认多轮扩展 |
| 生成 | 仅打印证据 + 朴素串答案 | 任意回答模型（可选） |

**跑官方 SAG2**：请用 [SAG-Benchmark](https://github.com/Zleap-AI/SAG-Benchmark) 的 `sag2`，并配置 OpenRouter / JEV 等密钥；桌面产品见 [SAG Wiki](https://github.com/Zleap-AI/SAG-Wiki)。

## 分词说明

SQLite FTS5 默认不按中文分词。本 Demo 把正文编成 **单字 + 双字（bigram）+ 英文/数字词** 再写入 FTS，避免引入 `jieba` 等依赖。正式系统会用更完整的分词与词项统计。
