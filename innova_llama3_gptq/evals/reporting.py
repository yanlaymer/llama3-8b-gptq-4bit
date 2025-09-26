"""Reporting utilities for evaluation results"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd


def create_results_summary(
    perplexity_results: Dict[str, Any],
    task_results: Optional[Dict[str, Any]] = None,
    latency_results: Optional[Dict[str, Any]] = None,
    model_info: Optional[Dict[str, Any]] = None,
    hardware_info: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create a comprehensive results summary

    Args:
        perplexity_results: Perplexity evaluation results
        task_results: Task evaluation results
        latency_results: Latency measurement results
        model_info: Model information
        hardware_info: Hardware information

    Returns:
        Comprehensive results dictionary
    """
    summary = {
        "timestamp": datetime.now().isoformat(),
        "model": model_info or {},
        "hardware": hardware_info or {},
        "perplexity": perplexity_results,
        "tasks": task_results or {},
        "latency": latency_results or {},
        "summary_metrics": {}
    }

    # Extract key metrics for summary
    if perplexity_results:
        ppl_values = []
        for dataset, metrics in perplexity_results.items():
            if isinstance(metrics, dict) and "perplexity" in metrics:
                ppl_values.append(metrics["perplexity"])
                summary["summary_metrics"][f"ppl_{dataset}"] = metrics["perplexity"]

        if ppl_values:
            summary["summary_metrics"]["avg_perplexity"] = sum(ppl_values) / len(ppl_values)

    if task_results:
        task_scores = []
        for task, score in task_results.items():
            if isinstance(score, (int, float)):
                task_scores.append(score)
                summary["summary_metrics"][f"task_{task}"] = score

        if task_scores:
            summary["summary_metrics"]["avg_task_score"] = sum(task_scores) / len(task_scores)

    if latency_results and "batch_1" in latency_results:
        summary["summary_metrics"]["latency_ms"] = latency_results["batch_1"]["avg_latency_ms"]
        summary["summary_metrics"]["throughput_tps"] = latency_results["batch_1"].get(
            "throughput_samples_per_sec", 0
        )

    return summary


def create_comparison_table(
    baseline_results: Dict[str, Any],
    quantized_results: Dict[str, Any],
    metrics_to_compare: Optional[list] = None
) -> pd.DataFrame:
    """
    Create a comparison table between baseline and quantized models

    Args:
        baseline_results: Baseline model results
        quantized_results: Quantized model results
        metrics_to_compare: List of metrics to compare

    Returns:
        Pandas DataFrame with comparison
    """
    if metrics_to_compare is None:
        metrics_to_compare = [
            "ppl_wikitext2",
            "ppl_c4",
            "task_hellaswag",
            "task_arc_easy",
            "task_mmlu",
            "latency_ms",
            "throughput_tps"
        ]

    data = []
    for metric in metrics_to_compare:
        baseline_value = baseline_results.get("summary_metrics", {}).get(metric, "N/A")
        quantized_value = quantized_results.get("summary_metrics", {}).get(metric, "N/A")

        if isinstance(baseline_value, (int, float)) and isinstance(quantized_value, (int, float)):
            delta = quantized_value - baseline_value
            delta_pct = (delta / baseline_value * 100) if baseline_value != 0 else 0
            data.append({
                "Metric": metric,
                "FP16": f"{baseline_value:.3f}" if isinstance(baseline_value, float) else baseline_value,
                "GPTQ": f"{quantized_value:.3f}" if isinstance(quantized_value, float) else quantized_value,
                "Delta": f"{delta:.3f}",
                "Delta%": f"{delta_pct:+.1f}%"
            })
        else:
            data.append({
                "Metric": metric,
                "FP16": baseline_value,
                "GPTQ": quantized_value,
                "Delta": "N/A",
                "Delta%": "N/A"
            })

    return pd.DataFrame(data)


def generate_markdown_report(
    results: Dict[str, Any],
    output_path: Optional[Path] = None
) -> str:
    """
    Generate a markdown report from results

    Args:
        results: Evaluation results
        output_path: Optional path to save the report

    Returns:
        Markdown string
    """
    report = ["# GPTQ Quantization Evaluation Report\n"]
    report.append(f"Generated: {results.get('timestamp', 'N/A')}\n")

    # Model Information
    if "model" in results:
        report.append("## Model Information\n")
        for key, value in results["model"].items():
            report.append(f"- **{key}**: {value}\n")

    # Hardware Information
    if "hardware" in results:
        report.append("\n## Hardware Configuration\n")
        for key, value in results["hardware"].items():
            report.append(f"- **{key}**: {value}\n")

    # Summary Metrics
    if "summary_metrics" in results:
        report.append("\n## Summary Metrics\n")
        report.append("| Metric | Value |\n")
        report.append("|--------|-------|\n")
        for key, value in results["summary_metrics"].items():
            if isinstance(value, float):
                report.append(f"| {key} | {value:.3f} |\n")
            else:
                report.append(f"| {key} | {value} |\n")

    # Perplexity Results
    if "perplexity" in results and results["perplexity"]:
        report.append("\n## Perplexity Results\n")
        for dataset, metrics in results["perplexity"].items():
            if isinstance(metrics, dict) and "perplexity" in metrics:
                report.append(f"- **{dataset}**: {metrics['perplexity']:.3f}\n")

    # Task Results
    if "tasks" in results and results["tasks"]:
        report.append("\n## Task Evaluation Results\n")
        for task, score in results["tasks"].items():
            if isinstance(score, (int, float)):
                report.append(f"- **{task}**: {score:.3f}\n")

    # Latency Results
    if "latency" in results and results["latency"]:
        report.append("\n## Latency Measurements\n")
        report.append("| Batch Size | Latency (ms) | Throughput (samples/s) |\n")
        report.append("|------------|-------------|------------------------|\n")
        for batch_key, metrics in results["latency"].items():
            if isinstance(metrics, dict):
                batch_size = batch_key.replace("batch_", "")
                latency = metrics.get("avg_latency_ms", "N/A")
                throughput = metrics.get("throughput_samples_per_sec", "N/A")
                if isinstance(latency, float):
                    latency = f"{latency:.2f}"
                if isinstance(throughput, float):
                    throughput = f"{throughput:.2f}"
                report.append(f"| {batch_size} | {latency} | {throughput} |\n")

    markdown = "\n".join(report)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(markdown)

    return markdown


def save_results(
    results: Dict[str, Any],
    output_dir: Path,
    format: str = "json"
) -> None:
    """
    Save evaluation results to file

    Args:
        results: Results to save
        output_dir: Output directory
        format: Output format (json, csv, markdown)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if format == "json":
        output_path = output_dir / f"metrics_{timestamp}.json"
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

    elif format == "csv" and "summary_metrics" in results:
        output_path = output_dir / f"metrics_{timestamp}.csv"
        df = pd.DataFrame([results["summary_metrics"]])
        df.to_csv(output_path, index=False)

    elif format == "markdown":
        output_path = output_dir / f"report_{timestamp}.md"
        generate_markdown_report(results, output_path)

    else:
        raise ValueError(f"Unsupported format: {format}")