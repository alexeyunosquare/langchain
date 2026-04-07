"""
Evaluation package for Agentic RAG.

This package provides comprehensive evaluation and benchmarking capabilities
for assessing the performance of the Agentic RAG system.

Phase 6: Evaluation and Benchmarking
- Metrics calculation (answer quality, retrieval accuracy, hallucination)
- Benchmark execution and result aggregation
- Comparative evaluation across configurations
"""

from .benchmarks import BenchmarkDataset, BenchmarkResults, BenchmarkRunner
from .evaluator import (AutomatedEvaluator, ComparativeEvaluator,
                        EvaluationConfig)
from .metrics import (AnswerMetrics, HallucinationMetrics, MetricsCalculator,
                      PerformanceMetrics, RetrievalMetrics)

__all__ = [
    # Metrics
    "AnswerMetrics",
    "RetrievalMetrics",
    "HallucinationMetrics",
    "PerformanceMetrics",
    "MetricsCalculator",
    # Benchmarks
    "BenchmarkDataset",
    "BenchmarkResults",
    "BenchmarkRunner",
    # Evaluator
    "EvaluationConfig",
    "AutomatedEvaluator",
    "ComparativeEvaluator",
]

__version__ = "0.1.0"
