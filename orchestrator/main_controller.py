# absa_training_pipeline.py
# 自动批量训练 + 验证 + 可视化 + 报告生成的主控入口

import os
import subprocess
import pandas as pd

def run_training():
    print("\n开始批量训练所有组合...")
    subprocess.run(["python", "controller_trainer.py"])

def run_evaluation():
    print("\n开始批量验证所有组合...")
    subprocess.run(["python", "controller_trainer.py"])

def run_plotting():
    print("\n开始绘制训练过程曲线与最终模型对比图...")
    subprocess.run(["python", "plot_log.py"], cwd="orchestrator")
    subprocess.run(["python", "visualize_results.py"], cwd="orchestrator")

def generate_markdown_report():
    print("\n开始生成实验总结Markdown报告...")
    final_results_path = os.path.join("orchestrator", "checkpoints", "final_results.csv")
    if not os.path.exists(final_results_path):
        print(f"找不到结果文件: {final_results_path}")
        return

    df = pd.read_csv(final_results_path)
    report = "# ABSA模型实验对比报告\n\n"
    report += "| Backbone | Finetune Strategy | BIO F1 | Sentiment ACC |\n"
    report += "|----------|-------------------|--------|----------------|\n"
    for _, row in df.iterrows():
        report += f"| {row['Backbone']} | {row['Finetune']} | {row['BIO F1']:.4f} | {row['Sentiment ACC']:.4f} |\n"
    with open("orchestrator/checkpoints/final_report.md", "w", encoding="utf-8") as f:
        f.write(report)
    print("报告已生成: checkpoints/final_report.md")

def main():
    run_training()
    run_evaluation()
    run_plotting()
    generate_markdown_report()

if __name__ == "__main__":
    main()