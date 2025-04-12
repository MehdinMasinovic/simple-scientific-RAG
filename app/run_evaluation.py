import os
import argparse
import subprocess
import time
from datetime import datetime


def run_command(command, description):
    """Run a shell command and print output."""
    print(f"\n{'=' * 80}\n{description}\n{'=' * 80}")
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())

    return process.poll()


def run_evaluation_workflow(questions_file, references_file,
                            llm_models=None, embedding_models=None, chunk_sizes=None,
                            max_workers=2, eval_model="llama3.2"):
    """Run the complete evaluation workflow."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create directories
    benchmark_dir = f"../data/benchmark_results_{timestamp}"
    eval_dir = f"../data/evaluation_results_{timestamp}"
    viz_dir = f"../data/visualizations_{timestamp}"

    os.makedirs(benchmark_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)

    # Step 1: Run benchmarks
    benchmark_cmd = f"python benchmark.py --questions {questions_file} --references {references_file} --output-dir {benchmark_dir} --max-workers {max_workers}"

    if llm_models:
        benchmark_cmd += f" --llm-models {' '.join(llm_models)}"

    if embedding_models:
        benchmark_cmd += f" --embedding-models {' '.join(embedding_models)}"

    if chunk_sizes:
        benchmark_cmd += f" --chunk-sizes {' '.join(map(str, chunk_sizes))}"

    run_command(benchmark_cmd, "Running benchmarks")

    # Step 2: Evaluate benchmarks
    eval_cmd = f"python evaluate_benchmarks.py --benchmark-dir {benchmark_dir} --output-dir {eval_dir} --eval-model {eval_model}"
    run_command(eval_cmd, "Evaluating benchmarks")

    # Step 3: Visualize results
    viz_cmd = f"python visualize_results.py --eval-dir {eval_dir} --output-dir {viz_dir}"
    run_command(viz_cmd, "Generating visualizations")

    print(f"\n{'=' * 80}")
    print(f"Evaluation workflow complete!")
    print(f"Benchmark results: {benchmark_dir}")
    print(f"Evaluation results: {eval_dir}")
    print(f"Visualizations: {viz_dir}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the complete RAG evaluation workflow")

    parser.add_argument('--questions', type=str, default='../data/sample_questions.txt',
                        help='Path to the questions file')

    parser.add_argument('--references', type=str, default='../data/sample_answers.txt',
                        help='Path to the reference answers file')

    parser.add_argument('--llm-models', type=str, nargs='+',
                        help='Specific LLM models to test')

    parser.add_argument('--embedding-models', type=str, nargs='+',
                        help='Specific embedding models to test')

    parser.add_argument('--chunk-sizes', type=int, nargs='+',
                        help='Specific chunk sizes to test')

    parser.add_argument('--max-workers', type=int, default=2,
                        help='Maximum number of parallel workers')

    parser.add_argument('--eval-model', type=str, default='llama3.2',
                        help='Model to use for evaluation')

    args = parser.parse_args()

    run_evaluation_workflow(
        questions_file=args.questions,
        references_file=args.references,
        llm_models=args.llm_models,
        embedding_models=args.embedding_models,
        chunk_sizes=args.chunk_sizes,
        max_workers=args.max_workers,
        eval_model=args.eval_model
    )