# Product direction: multi-asset opportunity screener and portfolio advisor

Status (2026-10-11): **direction approved by the owner on 2026-10-11** (OD1, OD2, OD4, OD5, OD6, OD7 in `vision_and_prd.md`, section 13); other decisions pending. Nothing in these documents is implemented. They change no research contract, universe, label, lockbox or point-in-time rule. Every such change below is listed as an owner decision.

| Document | What it answers |
|---|---|
| [`vision_and_prd.md`](vision_and_prd.md) | What the product is and is not, for whom, the capability tiers, the MVP, later capabilities, owner decisions and the first ticket |
| [`architecture.md`](architecture.md) | Three architectures and the recommended one, the layers, how absolute forecasts could be composed and when they stay unavailable, the asset-class and cross-cutting design questions, and the output contracts |
| [`roadmap.md`](roadmap.md) | The research and product tracks, their dependencies and acceptance gates |
| [`ux_blueprint.md`](ux_blueprint.md) | The three modes, screen concepts, evidence labels, first-class states, uncertainty display and the design language |
| [`risk_register.md`](risk_register.md) | Statistical, market-data, portfolio, legal and compliance, privacy and security, and product-design risks |

## How to read the capability tiers

Every capability in these documents carries one tier. The tier is a claim about evidence, not about effort.

Labels used across the documents: owner decisions OD1–OD15; ladder rungs L0–L4 and D1–D2; universe rules R1–R8; product tickets P*n*; research tickets RES-*n*; risks by category prefix (STAT, DATA, PORT, LEGAL, SEC, UX).

| Tier | Meaning | What the product may say |
|---|---|---|
| **NOW** | Exists in the repository today and is tested | Facts about the past and about the research pipeline |
| **RULES** | Can be built from transparent, pre-registered rules and historical data, with no forecast | "This is what the rule does" and "this is what happened historically under stated assumptions" |
| **VALIDATED** | Needs a model that has passed the validation ladder (`vision_and_prd.md`, "Validation ladder") on a point-in-time universe | Only the claim level the model reached, with its uncertainty and its evidence |
| **LONG-TERM** | Needs new asset-class contracts, licensed data, prospective evidence or a regulatory decision | Nothing until then |

## Verified starting point

- **Built:** a leakage-safe research framework.
  - Purged walk-forward validation (T1).
  - Diagnostics and dependence-aware inference (T2).
  - A read-only research dashboard (T3, T3.1).
  - Linear baselines with nested purged selection (T4, PR #47, merge pending).
- **T4 result:** on a four-name, hindsight-selected universe (INTC, MRVL, MU, NVDA), Ridge, Lasso and Elastic Net showed **no detectable out-of-sample skill**. See `docs/research/linear_models.md` (added by PR #47).
- **What this means:** it is an engineering and research foundation, **not** an investment-ready forecasting product. No current model output may be presented as a validated expected return, probability of gain, recommendation or portfolio forecast.
