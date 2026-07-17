# E7, E22-E26 消融实验：100 任务完整对比表

## 图例

| 缩写 | 含义 |
|------|------|
| `P` | PASS，无记忆注入 |
| `P+` | PASS，有记忆注入 |
| `F` | FAIL，无记忆注入 |
| `Fm` | FAIL，有记忆注入 |
| `F?` | FAIL，轨迹文件缺失 |
| `⬆E22:src(s)` | E22 改善，记忆来自 src，分数 s |
| `⬇E22:src(s)` | E22 退化，记忆来自 src，分数 s |

## 实验配置

| 参数 | E7（基线） | E22 | E23 | E24 | E25 | E26 |
|------|-----------|-----|-----|-----|-----|-----|
| **use_memory** | ❌ false | ✅ true | ✅ true | ✅ true | ✅ true | ✅ true |
| **use_anchor_in_embedding** | — | ❌ false | ❌ false | ❌ false | ❌ false | ❌ false |
| **alpha_semantic** | — | 0.70 | 0.50 | 0.60 | 0.50 | 0.40 |
| **alpha_cognitive** | — | 0.30 | 0.50 | 0.40 | 0.50 | **0.60** |
| **alpha_dual_task** | — | 0.70 | 0.70 | 0.70 | 0.70 | 0.70 |
| **top_k** | — | 1 | 3 | 2 | 1 | 1 |
| **top_n_candidates** | — | 10 | 10 | 10 | 10 | 10 |
| **score_threshold_floor** | — | 0.45 | 0.50 | **0.55** | 0.50 | 0.50 |
| **score_threshold_std** | — | 0.5 | 0.5 | **0.0** | 0.5 | 0.5 |
| **min_memories** | — | 0 | 0 | 0 | 0 | 0 |

> **E22 vs E23**: E22 偏语义（α_sem=0.70, α_cog=0.30），top_k=1，阈值 0.45；E23 平衡语义/认知（各 0.50），top_k=3，阈值 0.50
>
> **E23 vs E25**: 配置几乎相同（α_sem=0.50, α_cog=0.50, 阈值 0.50），唯一区别是 top_k（E23=3, E25=1）
>
> **E25 vs E26**: E26 进一步提高认知权重（α_cog=0.60）以测试 LLM 结构匹配是否优于 embedding 相似度
>
> **E24**: 最保守——高阈值（0.55）+ std=0.0，导致几乎不注入记忆（仅 5/500 次），**实质上等价于 E7 的重复实验**

## 100 任务对比表

| # | Rnd | Task | Repo | E7 | E22 | E23 | E24 | E25 | E26 | 分类 | 记忆详情 |
|---|-----|------|------|----|-----|-----|-----|-----|-----|------|----------|
| | | **🟢 稳定通过 (49个)** | | | | | | | | | |
| 1 | 2 | astropy__astropy-14508 | astropy | P | P | P | P | P | P | 稳定通过 | ⬆E22,E23,E24,E25,E26 |
| 2 | 3 | astropy__astropy-14539 | astropy | P | P | P | P | P | P | 稳定通过 | ⬆E22,E23,E24,E25,E26 |
| 3 | 4 | astropy__astropy-14995 | astropy | P | P | P | P | P | P | 稳定通过 | ⬆E22,E23,E24,E25,E26 |
| 4 | 5 | astropy__astropy-7166 | astropy | P | P | P | P | P | P | 稳定通过 | ⬆E22,E23,E24,E25,E26 |
| 5 | 10 | django__django-11211 | django | P | P | P | P | P | P | 稳定通过 | ⬆E22,E23,E24,E25,E26 |
| 6 | 11 | django__django-11292 | django | P | P | P | P | P+ | P | 稳定通过 | ⬆E25:django__django-11149(0.63)<br>⬆E22,E23,E24,E26 |
| 7 | 13 | django__django-11451 | django | P | P+ | P | P | P+ | P | 稳定通过 | ⬆E22:django__django-11211(0.46)<br>⬆E25:django__django-11292(0.53)<br>⬆E23,E24,E26 |
| 8 | 14 | django__django-11490 | django | P | P+ | P+ | P | P | P+ | 稳定通过 | ⬆E22:django__django-10554(0.68)<br>⬆E23:django__django-10554(0.79),django__django-10554(0.75),django__django-10554(0.75)<br>⬆E26:django__django-10554(0.81)<br>⬆E24,E25 |
| 9 | 18 | django__django-12125 | django | P | P+ | P | P | P | P+ | 稳定通过 | ⬆E22:django__django-11141(0.64)<br>⬆E26:django__django-11141(0.70)<br>⬆E23,E24,E25 |
| 10 | 19 | django__django-12304 | django | P | P | P+ | P | P | P+ | 稳定通过 | ⬆E23:django__django-12125(0.52)<br>⬆E26:django__django-11400(0.73)<br>⬆E22,E24,E25 |
| 11 | 21 | django__django-12419 | django | P | P | P+ | P | P | P | 稳定通过 | ⬆E23:django__django-12325(0.56),django__django-11451(0.51)<br>⬆E22,E24,E25,E26 |
| 12 | 24 | django__django-13343 | django | P | P+ | P+ | P | P+ | P+ | 稳定通过 | ⬆E22:django__django-11141(0.58)<br>⬆E23:django__django-12125(0.58)<br>⬆E25:django__django-12304(0.58)<br>⬆E26:django__django-11400(0.56)<br>⬆E24 |
| 13 | 25 | django__django-13363 | django | P | P | P | P | P+ | P | 稳定通过 | ⬆E25:django__django-12125(0.51)<br>⬆E22,E23,E24,E26 |
| 14 | 28 | django__django-13417 | django | P | P+ | P+ | P | P | P+ | 稳定通过 | ⬆E22:django__django-10554(0.62)<br>⬆E23:django__django-10554(0.53)<br>⬆E26:django__django-10554(0.65)<br>⬆E24,E25 |
| 15 | 32 | django__django-13933 | django | P | P+ | P | P | P | P+ | 稳定通过 | ⬆E22:django__django-12304(0.58)<br>⬆E26:django__django-11400(0.59)<br>⬆E23,E24,E25 |
| 16 | 33 | django__django-13964 | django | P | P | P+ | P | P | P | 稳定通过 | ⬆E23:django__django-11211(0.57)<br>⬆E22,E24,E25,E26 |
| 17 | 35 | django__django-14089 | django | P | P | P | P | P | P+ | 稳定通过 | ⬆E26:django__django-13933(0.61)<br>⬆E22,E23,E24,E25 |
| 18 | 36 | django__django-14311 | django | P | P+ | P | P | P+ | P | 稳定通过 | ⬆E22:django__django-12125(0.49)<br>⬆E25:django__django-11141(0.55)<br>⬆E23,E24,E26 |
| 19 | 38 | django__django-14915 | django | P | P | P | P | P+ | P | 稳定通过 | ⬆E25:django__django-13933(0.51)<br>⬆E22,E23,E24,E26 |
| 20 | 41 | django__django-15161 | django | P | P | P+ | P | P | P | 稳定通过 | ⬆E23:django__django-12125(0.70),django__django-12125(0.68)<br>⬆E22,E24,E25,E26 |
| 21 | 42 | django__django-15268 | django | P | P | P+ | P | P+ | P+ | 稳定通过 | ⬆E23:django__django-15161(0.55)<br>⬆E25:django__django-15161(0.57)<br>⬆E26:django__django-15022(0.60)<br>⬆E22,E24 |
| 22 | 45 | django__django-15375 | django | P | P+ | P | P | P+ | P | 稳定通过 | ⬆E22:django__django-13417(0.58)<br>⬆E25:django__django-11490(0.57)<br>⬆E23,E24,E26 |
| 23 | 46 | django__django-15499 | django | P | P+ | P+ | P | P+ | P+ | 稳定通过 | ⬆E22:django__django-15268(0.70)<br>⬆E23:django__django-15268(0.64)<br>⬆E25:django__django-15268(0.73)<br>⬆E26:django__django-15161(0.66)<br>⬆E24 |
| 24 | 47 | django__django-15503 | django | P | P | P+ | P | P+ | P+ | 稳定通过 | ⬆E23:django__django-15278(0.59),django__django-15278(0.57)<br>⬆E25:django__django-15278(0.59)<br>⬆E26:django__django-11211(0.65)<br>⬆E22,E24 |
| 25 | 50 | django__django-16493 | django | P | P+ | P+ | P | P+ | P+ | 稳定通过 | ⬆E22:django__django-13343(0.71)<br>⬆E23:django__django-13343(0.78),django__django-13343(0.75),django__django-13343(0.73)<br>⬆E25:django__django-13343(0.78)<br>⬆E26:django__django-13343(0.83)<br>⬆E24 |
| 26 | 52 | django__django-16612 | django | P | P | P+ | P | P | P | 稳定通过 | ⬆E23:django__django-15022(0.63)<br>⬆E22,E24,E25,E26 |
| 27 | 53 | matplotlib__matplotlib-20859 | matplotlib | P | P | P | P | P | P+ | 稳定通过 | ⬆E26:astropy__astropy-14539(0.65)<br>⬆E22,E23,E24,E25 |
| 28 | 56 | matplotlib__matplotlib-26113 | matplotlib | P | P | P+ | P | P | P | 稳定通过 | ⬆E23:matplotlib__matplotlib-24970(0.57),matplotlib__matplotlib-24970(0.53),matplotlib__matplotlib-24970(0.52)<br>⬆E22,E24,E25,E26 |
| 29 | 59 | pydata__xarray-3305 | pydata | P | P+ | P | P | P | P | 稳定通过 | ⬆E22:django__django-10554(0.49)<br>⬆E23,E24,E25,E26 |
| 30 | 60 | pydata__xarray-3677 | pydata | P | P+ | P | P | P | P | 稳定通过 | ⬆E22:pydata__xarray-3305(0.48)<br>⬆E23,E24,E25,E26 |
| 31 | 62 | pydata__xarray-4695 | pydata | P | P | P | P+ | P | P | 稳定通过 | ⬆E24:pydata__xarray-3677(0.60),(0.00)<br>⬆E22,E23,E25,E26 |
| 32 | 67 | pytest-dev__pytest-10081 | pytest-dev | P | P | P | P | P | P | 稳定通过 | ⬆E22,E23,E24,E25,E26 |
| 33 | 70 | pytest-dev__pytest-7432 | pytest-dev | P | P+ | P | P | P | P+ | 稳定通过 | ⬆E22:pytest-dev__pytest-10081(0.53)<br>⬆E26:pytest-dev__pytest-10081(0.61)<br>⬆E23,E24,E25 |
| 34 | 71 | pytest-dev__pytest-7571 | pytest-dev | P | P+ | P+ | P | P | P | 稳定通过 | ⬆E22:pytest-dev__pytest-10356(0.54)<br>⬆E23:pytest-dev__pytest-10081(0.59)<br>⬆E24,E25,E26 |
| 35 | 78 | scikit-learn__scikit-learn-14141 | scikit-learn | P | P | P | P | P | P | 稳定通过 | ⬆E22,E23,E24,E25,E26 |
| 36 | 81 | scikit-learn__scikit-learn-26323 | scikit-learn | P | P+ | P+ | P+ | P+ | P+ | 稳定通过 | ⬆E22:pydata__xarray-6992(0.49)<br>⬆E23:scikit-learn__scikit-learn-25102(0.52)<br>⬆E24:scikit-learn__scikit-learn-25102(0.56)<br>⬆E25:scikit-learn__scikit-learn-25102(0.58)<br>⬆E26:scikit-learn__scikit-learn-25102(0.55) |
| 37 | 83 | sphinx-doc__sphinx-7454 | sphinx-doc | P | P | P | P | P | P | 稳定通过 | ⬆E22,E23,E24,E25,E26 |
| 38 | 84 | sphinx-doc__sphinx-7889 | sphinx-doc | P | P+ | P | P | P | P+ | 稳定通过 | ⬆E22:sphinx-doc__sphinx-7454(0.53)<br>⬆E26:sphinx-doc__sphinx-7454(0.71)<br>⬆E23,E24,E25 |
| 39 | 86 | sphinx-doc__sphinx-9230 | sphinx-doc | P | P+ | P+ | P+ | P+ | P+ | 稳定通过 | ⬆E22:sphinx-doc__sphinx-7454(0.45)<br>⬆E23:sphinx-doc__sphinx-7454(0.63)<br>⬆E24:sphinx-doc__sphinx-9229(0.58)<br>⬆E25:sphinx-doc__sphinx-7454(0.65)<br>⬆E26:sphinx-doc__sphinx-7454(0.56) |
| 40 | 89 | sympy__sympy-13480 | sympy | P | P | P | F? | P | P | 稳定通过 | ⬆E22,E23,E25,E26 |
| 41 | 91 | sympy__sympy-14711 | sympy | P | P+ | P | P | P | P | 稳定通过 | ⬆E22:astropy__astropy-14995(0.54)<br>⬆E23,E24,E25,E26 |
| 42 | 93 | sympy__sympy-16450 | sympy | P | P | P | P | P | P | 稳定通过 | ⬆E22,E23,E24,E25,E26 |
| 43 | 94 | sympy__sympy-16766 | sympy | P | P | P+ | P+ | P+ | P+ | 稳定通过 | ⬆E23:sympy__sympy-14531(0.62)<br>⬆E24:sympy__sympy-16450(0.58)<br>⬆E25:sympy__sympy-14531(0.56)<br>⬆E26:sympy__sympy-14531(0.60)<br>⬆E22 |
| 44 | 96 | sympy__sympy-19495 | sympy | P | P | P | P | P+ | P | 稳定通过 | ⬆E25:matplotlib__matplotlib-26113(0.56)<br>⬆E22,E23,E24,E26 |
| 45 | 97 | sympy__sympy-20801 | sympy | P | P+ | P | P | P | P | 稳定通过 | ⬆E22:sympy__sympy-15017(0.61)<br>⬆E23,E24,E25,E26 |
| 46 | 99 | sympy__sympy-21847 | sympy | P | P | P | P | P | P+ | 稳定通过 | ⬆E26:sympy__sympy-16450(0.58)<br>⬆E22,E23,E24,E25 |
| 47 | 100 | sympy__sympy-22456 | sympy | P | P+ | P | P | P+ | P | 稳定通过 | ⬆E22:sympy__sympy-16450(0.47)<br>⬆E25:sympy__sympy-20801(0.51)<br>⬆E23,E24,E26 |
| 48 | 101 | sympy__sympy-23413 | sympy | P | P | P | P | P | P | 稳定通过 | ⬆E22,E23,E24,E25,E26 |
| 49 | 102 | sympy__sympy-24443 | sympy | P | P+ | P+ | P | P | P | 稳定通过 | ⬆E22:sympy__sympy-20801(0.49)<br>⬆E23:sympy__sympy-18199(0.64)<br>⬆E24,E25,E26 |
| | | **🔴 稳定失败(无记忆可用) (7个)** | | | | | | | | | |
| 50 | 1 | astropy__astropy-13398 | astropy | F | F | F | F | F | F | 稳定失败(无记忆可用) | ⬇E22,E23,E24,E25,E26 |
| 51 | 6 | django__django-10097 | django | F | F | F | F | F | F | 稳定失败(无记忆可用) | ⬇E22,E23,E24,E25,E26 |
| 52 | 7 | django__django-10554 | django | F | F | F | F | F | F | 稳定失败(无记忆可用) | ⬇E22,E23,E24,E25,E26 |
| 53 | 8 | django__django-11141 | django | F | F | F | F | F | F | 稳定失败(无记忆可用) | ⬇E22,E23,E24,E25,E26 |
| 54 | 40 | django__django-15098 | django | F | F | F | F | F | F | 稳定失败(无记忆可用) | ⬇E22,E23,E24,E25,E26 |
| 55 | 58 | psf__requests-2931 | psf | F | F | F | F | F | F | 稳定失败(无记忆可用) | ⬇E22,E23,E24,E25,E26 |
| 56 | 82 | sphinx-doc__sphinx-11510 | sphinx-doc | F | F | F | F | F | F | 稳定失败(无记忆可用) | ⬇E22,E23,E24,E25,E26 |
| | | **🔴 稳定失败(有记忆仍败) (11个)** | | | | | | | | | |
| 57 | 12 | django__django-11400 | django | F | F | Fm | F | F | F | 稳定失败(有记忆仍败) | ⬇E23:django__django-10554(0.63),django__django-11211(0.59),django__django-10554(0.59)<br>⬇E22,E24,E25,E26 |
| 58 | 20 | django__django-12325 | django | F | Fm | F | F | F | Fm | 稳定失败(有记忆仍败) | ⬇E22:django__django-11400(0.56)<br>⬇E26:django__django-12304(0.56)<br>⬇E23,E24,E25 |
| 59 | 34 | django__django-14034 | django | F | F | F | F | F | Fm | 稳定失败(有记忆仍败) | ⬇E26:django__django-13933(0.66)<br>⬇E22,E23,E24,E25 |
| 60 | 55 | matplotlib__matplotlib-25479 | matplotlib | F | Fm | F | F | F | Fm | 稳定失败(有记忆仍败) | ⬇E22:matplotlib__matplotlib-24970(0.47)<br>⬇E26:matplotlib__matplotlib-24970(0.80)<br>⬇E23,E24,E25 |
| 61 | 63 | pydata__xarray-6992 | pydata | F | F | Fm | F | Fm | F | 稳定失败(有记忆仍败) | ⬇E23:pydata__xarray-3677(0.52)<br>⬇E25:pydata__xarray-3677(0.56)<br>⬇E22,E24,E26 |
| 62 | 64 | pylint-dev__pylint-4604 | pylint-dev | F | Fm | F | F | F | Fm | 稳定失败(有记忆仍败) | ⬇E22:django__django-11141(0.47)<br>⬇E26:django__django-11141(0.71)<br>⬇E23,E24,E25 |
| 63 | 66 | pylint-dev__pylint-7080 | pylint-dev | F | Fm | Fm | F? | Fm | F | 稳定失败(有记忆仍败) | ⬇E22:pylint-dev__pylint-6528(0.77)<br>⬇E23:pylint-dev__pylint-6528(0.85),pylint-dev__pylint-6528(0.70),pylint-dev__pylint-6528(0.82)<br>⬇E25:pylint-dev__pylint-6528(0.84)<br>⬇E26 |
| 64 | 68 | pytest-dev__pytest-10356 | pytest-dev | F | Fm | Fm | F | F | F | 稳定失败(有记忆仍败) | ⬇E22:pytest-dev__pytest-10081(0.50)<br>⬇E23:pytest-dev__pytest-10081(0.57)<br>⬇E24,E25,E26 |
| 65 | 85 | sphinx-doc__sphinx-9229 | sphinx-doc | F | Fm | Fm | F | Fm | Fm | 稳定失败(有记忆仍败) | ⬇E22:sphinx-doc__sphinx-7454(0.55)<br>⬇E23:sphinx-doc__sphinx-7454(0.56),sphinx-doc__sphinx-7454(0.52)<br>⬇E25:sphinx-doc__sphinx-7454(0.54)<br>⬇E26:sphinx-doc__sphinx-7454(0.52)<br>⬇E24 |
| 66 | 88 | sphinx-doc__sphinx-9711 | sphinx-doc | F | Fm | Fm | F | F | F | 稳定失败(有记忆仍败) | ⬇E22:sphinx-doc__sphinx-9229(0.50)<br>⬇E23:sphinx-doc__sphinx-9230(0.57),sphinx-doc__sphinx-9230(0.57)<br>⬇E24,E25,E26 |
| 67 | 95 | sympy__sympy-18199 | sympy | F | Fm | F | F | F | Fm | 稳定失败(有记忆仍败) | ⬇E22:astropy__astropy-14995(0.48)<br>⬇E26:sympy__sympy-14711(0.57)<br>⬇E23,E24,E25 |
| | | **🟢 记忆改善 (1个)** | | | | | | | | | |
| 68 | 27 | django__django-13406 | django | F | Fm | P+ | F | Fm | Fm | 记忆改善 | ⬇E22:django__django-10554(0.55)<br>⬇E25:django__django-11490(0.58)<br>⬇E26:django__django-10554(0.57)<br>⬇E24<br>⬆E23:django__django-10554(0.71),django__django-10554(0.61),django__django-11490(0.66) |
| | | **🟡 随机改善 (2个)** | | | | | | | | | |
| 69 | 15 | django__django-11532 | django | F | P | P | P | P | P | 随机改善 | ⬆E22,E23,E24,E25,E26 |
| 70 | 57 | mwaskom__seaborn-3069 | mwaskom | F | F | P | P | P | P | 随机改善 | ⬇E22<br>⬆E23,E24,E25,E26 |
| | | **🔵 混合波动 (30个)** | | | | | | | | | |
| 71 | 9 | django__django-11149 | django | P | P | P | P | F | P | 混合波动 | ⬇E25<br>⬆E22,E23,E24,E26 |
| 72 | 16 | django__django-11734 | django | P | Fm | P+ | F | P+ | P | 混合波动 | ⬇E22:django__django-10554(0.46)<br>⬇E24<br>⬆E23:django__django-10554(0.56),django__django-10554(0.50)<br>⬆E25:django__django-10554(0.69)<br>⬆E26 |
| 73 | 17 | django__django-11848 | django | P | P | P | P | F | P | 混合波动 | ⬇E25<br>⬆E22,E23,E24,E26 |
| 74 | 22 | django__django-13112 | django | F | P | F | P | P+ | P+ | 混合波动 | ⬇E23<br>⬆E25:django__django-12125(0.65)<br>⬆E26:django__django-11211(0.57)<br>⬆E22,E24 |
| 75 | 23 | django__django-13158 | django | F | P+ | P+ | P | P+ | P+ | 混合波动 | ⬆E22:django__django-10554(0.60)<br>⬆E23:django__django-10554(0.72),django__django-10554(0.69),django__django-11490(0.62)<br>⬆E25:django__django-10554(0.61)<br>⬆E26:django__django-10554(0.68)<br>⬆E24 |
| 76 | 26 | django__django-13401 | django | F | P | P+ | P | F | P+ | 混合波动 | ⬇E25<br>⬆E23:django__django-10554(0.59),django__django-12325(0.54)<br>⬆E26:django__django-13343(0.55)<br>⬆E22,E24 |
| 77 | 29 | django__django-13512 | django | P | Fm | Fm | P | F | Fm | 混合波动 | ⬇E22:django__django-11532(0.51)<br>⬇E23:django__django-11532(0.67),django__django-11532(0.63)<br>⬇E26:django__django-11532(0.60)<br>⬇E25<br>⬆E24 |
| 78 | 30 | django__django-13551 | django | P | P | P | P | P | F | 混合波动 | ⬇E26<br>⬆E22,E23,E24,E25 |
| 79 | 31 | django__django-13741 | django | P | P+ | P | P | P | F | 混合波动 | ⬇E26<br>⬆E22:django__django-13551(0.53)<br>⬆E23,E24,E25 |
| 80 | 37 | django__django-14349 | django | P | P+ | Fm | P | P+ | P+ | 混合波动 | ⬇E23:django__django-10097(0.70),django__django-10097(0.58)<br>⬆E22:django__django-10097(0.59)<br>⬆E25:django__django-10097(0.71)<br>⬆E26:django__django-10097(0.63)<br>⬆E24 |
| 81 | 39 | django__django-15022 | django | F | P+ | P | F | P+ | Fm | 混合波动 | ⬇E26:django__django-11400(0.58)<br>⬇E24<br>⬆E22:django__django-11400(0.47)<br>⬆E25:django__django-11400(0.61)<br>⬆E23 |
| 82 | 43 | django__django-15278 | django | F | P+ | P | P | P | P | 混合波动 | ⬆E22:django__django-13112(0.48)<br>⬆E23,E24,E25,E26 |
| 83 | 44 | django__django-15280 | django | F | P+ | P | P | P+ | P | 混合波动 | ⬆E22:django__django-11211(0.46)<br>⬆E25:django__django-11400(0.56)<br>⬆E23,E24,E26 |
| 84 | 48 | django__django-16032 | django | F | P+ | P+ | P | P+ | P+ | 混合波动 | ⬆E22:django__django-10554(0.57)<br>⬆E23:django__django-15375(0.66),django__django-10554(0.62),django__django-10554(0.56)<br>⬆E25:django__django-15280(0.62)<br>⬆E26:django__django-15375(0.57)<br>⬆E24 |
| 85 | 49 | django__django-16100 | django | P | Fm | P+ | F | P | P | 混合波动 | ⬇E22:django__django-11734(0.47)<br>⬇E24<br>⬆E23:django__django-15022(0.59)<br>⬆E25,E26 |
| 86 | 51 | django__django-16560 | django | P | P | P+ | F | P+ | P+ | 混合波动 | ⬇E24<br>⬆E23:django__django-13933(0.52)<br>⬆E25:django__django-15499(0.51)<br>⬆E26:django__django-13933(0.62)<br>⬆E22 |
| 87 | 54 | matplotlib__matplotlib-24970 | matplotlib | P | P | P | P | Fm | P | 混合波动 | ⬇E25:matplotlib__matplotlib-20859(0.61)<br>⬆E22,E23,E24,E26 |
| 88 | 61 | pydata__xarray-4687 | pydata | F | P | P+ | F | P+ | P+ | 混合波动 | ⬇E24<br>⬆E23:pydata__xarray-3305(0.73),pydata__xarray-3305(0.70),pydata__xarray-3305(0.71)<br>⬆E25:pydata__xarray-3305(0.73)<br>⬆E26:pydata__xarray-3305(0.74)<br>⬆E22 |
| 89 | 65 | pylint-dev__pylint-6528 | pylint-dev | P | P | P | F | P | P+ | 混合波动 | ⬇E24<br>⬆E26:django__django-14311(0.59)<br>⬆E22,E23,E25 |
| 90 | 69 | pytest-dev__pytest-6197 | pytest-dev | F | P+ | Fm | P | F | Fm | 混合波动 | ⬇E23:pytest-dev__pytest-10356(0.72),pytest-dev__pytest-10356(0.68),pytest-dev__pytest-10081(0.50)<br>⬇E26:pytest-dev__pytest-10081(0.58)<br>⬇E25<br>⬆E22:pytest-dev__pytest-10081(0.56)<br>⬆E24 |
| 91 | 74 | scikit-learn__scikit-learn-10297 | scikit-learn | P | P | P | P | P+ | Fm | 混合波动 | ⬇E26:pydata__xarray-3305(0.52)<br>⬆E25:pydata__xarray-3305(0.53)<br>⬆E22,E23,E24 |
| 92 | 75 | scikit-learn__scikit-learn-13124 | scikit-learn | P | P | P | P | F | Fm | 混合波动 | ⬇E26:scikit-learn__scikit-learn-10297(0.56)<br>⬇E25<br>⬆E22,E23,E24 |
| 93 | 76 | scikit-learn__scikit-learn-13142 | scikit-learn | P | P | P | P | F | P | 混合波动 | ⬇E25<br>⬆E22,E23,E24,E26 |
| 94 | 77 | scikit-learn__scikit-learn-13328 | scikit-learn | P | Fm | P+ | P | P | P+ | 混合波动 | ⬇E22:scikit-learn__scikit-learn-13142(0.47)<br>⬆E23:scikit-learn__scikit-learn-13142(0.66)<br>⬆E26:scikit-learn__scikit-learn-13142(0.53)<br>⬆E24,E25 |
| 95 | 79 | scikit-learn__scikit-learn-25102 | scikit-learn | P | P+ | P+ | F | P+ | P | 混合波动 | ⬇E24<br>⬆E22:scikit-learn__scikit-learn-13124(0.46)<br>⬆E23:pydata__xarray-3677(0.64),pydata__xarray-3677(0.62),pydata__xarray-3305(0.54)<br>⬆E25:scikit-learn__scikit-learn-10297(0.56)<br>⬆E26 |
| 96 | 80 | scikit-learn__scikit-learn-25973 | scikit-learn | P | P | P | F | P | P | 混合波动 | ⬇E24<br>⬆E22,E23,E25,E26 |
| 97 | 87 | sphinx-doc__sphinx-9258 | sphinx-doc | P | Fm | P+ | P+ | P+ | P+ | 混合波动 | ⬇E22:sphinx-doc__sphinx-9230(0.66)<br>⬆E23:sphinx-doc__sphinx-9230(0.71),sphinx-doc__sphinx-9230(0.67)<br>⬆E24:sphinx-doc__sphinx-9230(0.67),sphinx-doc__sphinx-9230(0.62)<br>⬆E25:sphinx-doc__sphinx-7454(0.64)<br>⬆E26:sphinx-doc__sphinx-9230(0.76) |
| 98 | 90 | sympy__sympy-14531 | sympy | P | P | Fm | P | P+ | P+ | 混合波动 | ⬇E23:pydata__xarray-3305(0.65),pydata__xarray-3305(0.63),pydata__xarray-3305(0.62)<br>⬆E25:pydata__xarray-3305(0.67)<br>⬆E26:pydata__xarray-6992(0.71)<br>⬆E22,E24 |
| 99 | 92 | sympy__sympy-15017 | sympy | P | P | Fm | P | P | F | 混合波动 | ⬇E23:sympy__sympy-14711(0.70),sympy__sympy-14711(0.60),sympy__sympy-14711(0.65)<br>⬇E26<br>⬆E22,E24,E25 |
| 100 | 98 | sympy__sympy-21379 | sympy | P | P+ | F | P | P | P | 混合波动 | ⬇E23<br>⬆E22:sympy__sympy-16450(0.67)<br>⬆E24,E25,E26 |

---

## 统计摘要

| 分类 | 数量 | 占比 |
|------|------|------|
| 🟢 稳定通过 | 49 | 49% |
| 🔴 稳定失败(无记忆可用) | 7 | 7% |
| 🔴 稳定失败(有记忆仍败) | 11 | 11% |
| 🟢 记忆改善 | 1 | 1% |
| 🟠 记忆退化 | 0 | 0% |
| 🟡 随机改善 | 2 | 2% |
| 🟡 随机退化 | 0 | 0% |
| 🔵 混合波动 | 30 | 30% |
| **合计** | **100** | **100%** |

- 稳定任务: 67 个 (67%)
- 记忆可归因: 1 个 (1%)
- 随机性可归因: 2 个 (2%)
- 混合/无法归因: 30 个 (30%)

---

## P/P+/F/Fm 完整统计

> **P** = PASS 无记忆注入 &nbsp;|&nbsp; **P+** = PASS 有记忆注入 &nbsp;|&nbsp; **F** = FAIL 无记忆注入 &nbsp;|&nbsp; **Fm** = FAIL 有记忆注入

### 各实验对比（100 任务 × 5 实验 = 500 次）

| | E7 | E22 | E23 | E24 | E25 | E26 | 合计 |
|---|---|---|---|---|---|---|---|
| **P** (PASS 无记忆) | 70 | 44 | 46 | 67 | 44 | 44 | **245** |
| **P+** (PASS 有记忆) | — | 31 | 29 | 5 | 29 | 29 | **123** |
| **F** (FAIL 无记忆) | 30 | 11 | 14 | 26 | 22 | 15 | **88** |
| **Fm** (FAIL 有记忆) | — | 14 | 11 | 0 | 5 | 12 | **42** |
| **F?** (轨迹缺失) | — | 0 | 0 | 2 | 0 | 0 | **2** |
| **合计** | **100** | **100** | **100** | **100** | **100** | **100** | **500** |
| **Pass%** | **70%** | **75%** | **75%** | **72%** | **73%** | **73%** | **73.6%** |
| **有记忆改善率** (P+/(P++Fm)) | — | 69% | 72% | 100% | 85% | 71% | **75%** |

### 有记忆 vs 无记忆 Pass 率对比

| | 次数 | 占比 | Pass 率 | Fail 率 |
|---|---|---|---|---|
| **有记忆行为** (P+ + Fm) | 165 | 33% | **75%** (123/165) | **25%** (42/165) |
| **无记忆行为** (P + F) | 333 | 67% | **74%** (245/333) | **26%** (88/333) |

> ⚠️ **有记忆时 Pass 率 75%，无记忆时 Pass 率 74%——几乎没有差异。** 记忆注入没有系统性提高成功率。E24 最有说服力：仅注入 5 次记忆，全靠 LLM 裸跑也拿了 72%，与其他实验相差无几。

### 按分类统计 P/P+/F/Fm（仅 E22-E26）

| 分类 | 任务 | P | P+ | F | Fm | 合计 | Pass% |
|------|------|---|---|---|---|---|---|---|
| 🟢 稳定通过 | 49 | 167 | 77 | 0 | 0 | 245 | **99.6%** |
| 🔴 稳定失败(无记忆) | 7 | 0 | 0 | 35 | 0 | 35 | 0% |
| 🔴 稳定失败(有记忆) | 11 | 0 | 0 | 31 | 23 | 55 | 0% |
| 🟢 记忆改善 | 1 | 0 | 1 | 1 | 3 | 5 | 20% |
| 🟡 随机改善 | 2 | 9 | 0 | 1 | 0 | 10 | 90% |
| 🔵 混合波动 | 30 | 69 | 45 | 20 | 16 | 150 | 76% |

### 核心结论

1. **有记忆 vs 无记忆的 Pass 率几乎相同**（75% vs 74%），记忆系统没有系统性提升
2. **E24 是关键对照**：阈值设为 0.55 导致几乎不注入记忆（仅 5/500 = 1% 有记忆），纯靠 LLM → **72%**。E22/E23/E25/E26 注入了更多记忆 → **73-75%**。差仅 1-3pp
3. **记忆的改善和退化相互抵消**：P+ 有 123 次（改善），Fm 有 42 次（退化），净效应 ≈ 0
4. **67% 的任务完全稳定**，不因实验配置变化而改变结果
5. **LLM 随机波动是主要噪声源**：30 个混合波动任务贡献了 150 次实验中的 114 PASS + 36 FAIL，波动幅度远超记忆信号