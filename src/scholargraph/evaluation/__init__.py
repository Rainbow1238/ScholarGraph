"""评测：比较不同系统配置生成的综述，量化引用的可信度、内容覆盖和成本。

    python -m scholargraph.evaluation run        # 跑评测（可中断，再次运行会续跑）
    python -m scholargraph.evaluation summary    # 汇总成对比表
    python -m scholargraph.evaluation sample     # 抽样导出，供人工标注
    python -m scholargraph.evaluation agreement  # 计算自动评委与人工标注的一致性

建议的阅读顺序：`dataset.py` → `systems.py` → `baselines.py` → `judge.py` → `runner.py` → `summary.py`。
"""
