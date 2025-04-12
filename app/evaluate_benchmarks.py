import os
import json
import argparse
from pathlib import Path
from datetime import datetime
import ollama
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric
)
from templates.answer_relevancy_template import AnswerRelevancyTemplate
from templates.contextual_precision_template import ContextualPrecisionTemplate
from templates.contextual_recall_template import ContextualRecallTemplate
from templates.contextual_relevancy_template import ContextualRelevancyTemplate
from templates.faithfulness_template import FaithfulnessTemplate


# Custom Ollama model for evaluation
class CustomOllamaModel(DeepEvalBaseLLM):
    def __init__(self, model_name: str = "llama3.2"):
        self.model_name = model_name

    def load_model(self):
        return self.model_name

    def generate(self, prompt: str) -> str:
        response = ollama.generate(
            model=self.model_name,
            prompt=prompt,
            options={'num_predict': 200, 'temperature': 0}
        )
        return response['response']

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return f"Ollama {self.model_name}"


def initialize_metrics(eval_model):
    """Initialize all evaluation metrics with the specified model."""
    metrics = {
        "answer_relevancy": AnswerRelevancyMetric(
            model=eval_model,
            threshold=0.7,
            evaluation_template=AnswerRelevancyTemplate,
            strict_mode=True
        ),
        "contextual_recall": ContextualRecallMetric(
            threshold=0.7,
            model=eval_model,
            include_reason=True,
            evaluation_template=ContextualRecallTemplate,
            strict_mode=True
        ),
        "contextual_precision": ContextualPrecisionMetric(
            threshold=0.7,
            model=eval_model,
            include_reason=True,
            evaluation_template=ContextualPrecisionTemplate,
            strict_mode=True
        ),
        "faithfulness": FaithfulnessMetric(
            threshold=0.7,
            model=eval_model,
            include_reason=True,
            evaluation_template=FaithfulnessTemplate,
            strict_mode=True
        ),
        "contextual_relevancy": ContextualRelevancyMetric(
            threshold=0.7,
            model=eval_model,
            include_reason=True,
            evaluation_template=ContextualRelevancyTemplate,
            strict_mode=True
        )
    }
    return metrics


def evaluate_result_file(result_file, eval_model, output_dir):
    """Evaluate a single benchmark result file."""
    # Load the benchmark results
    with open(result_file, 'r', encoding='utf-8') as f:
        benchmark_results = json.load(f)

    # Initialize metrics
    metrics = initialize_metrics(eval_model)

    # Process each benchmark result
    evaluation_results = []

    for result in benchmark_results:
        test_case = LLMTestCase(
            input=result["question"],
            actual_output=result["answer"],
            expected_output=result["reference"],
            retrieval_context=[result["context"]]
        )

        metrics_result = {
            "question": result["question"],
            "answer": result["answer"],
            "reference": result["reference"],
            "context": result["context"],
            "config": result["config"],
            "metrics": {}
        }

        # Measure all metrics
        for metric_name, metric in metrics.items():
            try:
                metric.measure(test_case)
                metrics_result["metrics"][metric_name] = {
                    "score": metric.score,
                    "reason": metric.reason
                }
            except Exception as e:
                print(f"Error measuring {metric_name}: {str(e)}")
                metrics_result["metrics"][metric_name] = {
                    "score": None,
                    "reason": f"Error: {str(e)}"
                }

        evaluation_results.append(metrics_result)

    # Save the evaluation results
    file_basename = os.path.basename(result_file)
    output_file = os.path.join(output_dir, f"evaluation_{file_basename}")

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(evaluation_results, f, indent=2, ensure_ascii=False)

    print(f"Evaluation results saved to {output_file}")
    return output_file


def evaluate_benchmarks(benchmark_dir, output_dir, eval_model_name="llama3.2"):
    """Evaluate all benchmark results in the specified directory."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Initialize evaluation model
    eval_model = CustomOllamaModel(model_name=eval_model_name)

    # Find all benchmark result files
    benchmark_files = list(Path(benchmark_dir).glob("results_*.json"))

    if not benchmark_files:
        print(f"No benchmark result files found in {benchmark_dir}")
        return

    evaluated_files = []

    # Evaluate each benchmark file
    for benchmark_file in benchmark_files:
        print(f"Evaluating {benchmark_file}...")
        evaluated_file = evaluate_result_file(benchmark_file, eval_model, output_dir)
        evaluated_files.append(evaluated_file)

    # Create a summary file
    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_evaluated_files": len(evaluated_files),
        "evaluation_model": eval_model_name,
        "evaluated_files": evaluated_files
    }

    summary_file = os.path.join(output_dir, f"evaluation_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print(f"Evaluation summary saved to {summary_file}")
    return summary_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RAG benchmark results")

    parser.add_argument('--benchmark-dir', type=str, default='../data/benchmark_results',
                        help='Directory containing benchmark results')

    parser.add_argument('--output-dir', type=str, default='../data/evaluation_results',
                        help='Directory to store evaluation results')

    parser.add_argument('--eval-model', type=str, default='llama3.2',
                        help='Model to use for evaluation')

    args = parser.parse_args()

    evaluate_benchmarks(
        benchmark_dir=args.benchmark_dir,
        output_dir=args.output_dir,
        eval_model_name=args.eval_model
    )