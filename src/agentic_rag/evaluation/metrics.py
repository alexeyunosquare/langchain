"""
Metrics calculation for Agentic RAG evaluation.

This module provides comprehensive metrics for evaluating RAG system performance,
including answer quality, retrieval accuracy, hallucination detection, and
system efficiency metrics.
"""

import re
import string
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
from langchain_core.documents import Document as LangChainDocument

from ..state import Document


@dataclass
class AnswerMetrics:
    """
    Metrics for answer quality assessment.

    Attributes:
        exact_match: Percentage of answers that exactly match ground truth
        f1_score: Harmonic mean of precision and recall for word overlap
        rouge_l_f1: ROUGE-L F1 score for longest common subsequence
        bleu_score: BLEU score for n-gram overlap
        semantic_similarity: Cosine similarity between answer embeddings
        llm_judge_score: Score from LLM-as-judge evaluation (0-1)
        relevance_score: How relevant the answer is to the query
    """

    exact_match: float = 0.0
    f1_score: float = 0.0
    rouge_l_f1: float = 0.0
    bleu_score: float = 0.0
    semantic_similarity: float = 0.0
    llm_judge_score: float = 0.0
    relevance_score: float = 0.0

    @property
    def overall_score(self) -> float:
        """Calculate overall answer quality score (weighted average)."""
        weights = {
            "exact_match": 0.1,
            "f1_score": 0.3,
            "rouge_l_f1": 0.2,
            "llm_judge_score": 0.4,
        }
        return sum(
            getattr(self, k, 0.0) * v
            for k, v in weights.items()
        )

    def __str__(self) -> str:
        return (
            f"AnswerMetrics(exact_match={self.exact_match:.3f}, "
            f"f1={self.f1_score:.3f}, rouge_l={self.rouge_l_f1:.3f}, "
            f"overall={self.overall_score:.3f})"
        )


@dataclass
class RetrievalMetrics:
    """
    Metrics for retrieval quality assessment.

    Attributes:
        precision_at_k: Fraction of retrieved documents that are relevant
        recall_at_k: Fraction of relevant documents that are retrieved
        mrr: Mean Reciprocal Rank of first relevant document
        ndcg_at_k: Normalized Discounted Cumulative Gain at k
        map: Mean Average Precision across queries
        hits_at_k: Number of relevant documents in top-k results
    """

    precision_at_k: float = 0.0
    recall_at_k: float = 0.0
    mrr: float = 0.0
    ndcg_at_k: float = 0.0
    map_score: float = 0.0
    hits_at_k: int = 0
    k: int = 10

    @property
    def f1_retrieval(self) -> float:
        """Calculate F1 score for retrieval (harmonic mean)."""
        if self.precision_at_k + self.recall_at_k == 0:
            return 0.0
        return 2 * (
            self.precision_at_k * self.recall_at_k
        ) / (self.precision_at_k + self.recall_at_k)

    def __str__(self) -> str:
        return (
            f"RetrievalMetrics(p@{self.k}={self.precision_at_k:.3f}, "
            f"r@{self.k}={self.recall_at_k:.3f}, mrr={self.mrr:.3f}, "
            f"f1={self.f1_retrieval:.3f})"
        )


@dataclass
class HallucinationMetrics:
    """
    Metrics for hallucination detection and fact consistency.

    Attributes:
        fact_consistency_score: Percentage of facts supported by documents
        contradiction_rate: Frequency of statements contradicting sources
        unsubstantiated_claims: Count of claims without document support
        hallucination_detected: Boolean flag for significant hallucination
        hallucination_severity: Severity score (0-1, higher = more severe)
        faithfulness_score: How faithful the answer is to the context
    """

    fact_consistency_score: float = 0.0
    contradiction_rate: float = 0.0
    unsubstantiated_claims: int = 0
    hallucination_detected: bool = False
    hallucination_severity: float = 0.0
    faithfulness_score: float = 0.0

    @property
    def overall_hallucination_score(self) -> float:
        """Calculate overall hallucination score (lower is better)."""
        return (
            (1 - self.fact_consistency_score)
            + self.contradiction_rate
            + (self.unsubstantiated_claims * 0.1)
        ) / 3

    def __str__(self) -> str:
        return (
            f"HallucinationMetrics(fact_consistency={self.fact_consistency_score:.3f}, "
            f"faithfulness={self.faithfulness_score:.3f}, "
            f"hallucination_detected={self.hallucination_detected}, "
            f"severity={self.hallucination_severity:.3f})"
        )


@dataclass
class PerformanceMetrics:
    """
    Metrics for system performance and efficiency.

    Attributes:
        total_latency_ms: Total execution time in milliseconds
        retrieval_latency_ms: Time spent on document retrieval
        generation_latency_ms: Time spent on answer generation
        documents_retrieved: Number of documents retrieved
        search_iterations: Number of search iterations performed
        token_usage: Number of tokens consumed
        cost_estimate: Estimated cost in dollars
    """

    total_latency_ms: float = 0.0
    retrieval_latency_ms: float = 0.0
    generation_latency_ms: float = 0.0
    documents_retrieved: int = 0
    search_iterations: int = 0
    token_usage: int = 0
    cost_estimate: float = 0.0

    @property
    def efficiency_score(self) -> float:
        """Calculate efficiency score (lower latency and tokens = higher score)."""
        if self.total_latency_ms <= 0 or self.token_usage <= 0:
            return 0.0
        latency_score = max(0, 1 - (self.total_latency_ms / 5000))
        token_score = max(0, 1 - (self.token_usage / 10000))
        return (latency_score + token_score) / 2

    def __str__(self) -> str:
        return (
            f"PerformanceMetrics(latency={self.total_latency_ms:.1f}ms, "
            f"docs={self.documents_retrieved}, iterations={self.search_iterations}, "
            f"tokens={self.token_usage}, efficiency={self.efficiency_score:.3f})"
        )


class MetricsCalculator:
    """
    Calculate comprehensive evaluation metrics for RAG responses.
    """

    def __init__(self, llm_model: Optional[str] = None):
        """Initialize metrics calculator."""
        self.llm_model = llm_model

    def calculate_answer_metrics(
        self,
        predicted: str,
        ground_truth: str,
        query: Optional[str] = None,
    ) -> AnswerMetrics:
        """Calculate answer quality metrics."""
        pred_normalized = self._normalize_text(predicted)
        gt_normalized = self._normalize_text(ground_truth)

        exact_match = 1.0 if pred_normalized == gt_normalized else 0.0
        f1_score = self._calculate_f1_score(pred_normalized, gt_normalized)
        rouge_l_f1 = self._calculate_rouge_l_f1(pred_normalized, gt_normalized)
        semantic_similarity = self._calculate_semantic_similarity(
            pred_normalized, gt_normalized
        )

        return AnswerMetrics(
            exact_match=exact_match,
            f1_score=f1_score,
            rouge_l_f1=rouge_l_f1,
            bleu_score=0.0,
            semantic_similarity=semantic_similarity,
            llm_judge_score=0.0,
            relevance_score=0.5,
        )

    def calculate_retrieval_metrics(
        self,
        retrieved_docs: List[Union[Document, LangChainDocument]],
        gold_docs: List[Union[Document, LangChainDocument]],
        top_k: int = 10,
    ) -> RetrievalMetrics:
        """Calculate retrieval quality metrics."""
        if not retrieved_docs or not gold_docs:
            return RetrievalMetrics(k=top_k)

        retrieved_ids = set(
            doc.metadata.get("id", i)
            for i, doc in enumerate(retrieved_docs[:top_k])
        )
        gold_ids = set(
            doc.metadata.get("id", i)
            for i, doc in enumerate(gold_docs)
        )

        hits = len(retrieved_ids & gold_ids)
        precision = hits / len(retrieved_ids) if retrieved_ids else 0.0
        recall = hits / len(gold_ids) if gold_ids else 0.0
        mrr = 0.0
        if retrieved_ids & gold_ids:
            first_hit_rank = None
            for idx, rid in enumerate(retrieved_ids):
                if rid in gold_ids:
                    first_hit_rank = idx + 1
                    break
            if first_hit_rank:
                mrr = 1.0 / first_hit_rank

        ndcg = self._calculate_ndcg(retrieved_docs, gold_docs, top_k)

        return RetrievalMetrics(
            precision_at_k=precision,
            recall_at_k=recall,
            mrr=mrr,
            ndcg_at_k=ndcg,
            map_score=0.0,
            hits_at_k=hits,
            k=top_k,
        )

    def calculate_hallucination_metrics(
        self,
        answer: str,
        documents: List[Union[Document, LangChainDocument]],
    ) -> HallucinationMetrics:
        """Calculate hallucination detection metrics."""
        if not documents:
            return HallucinationMetrics(
                hallucination_detected=True,
                hallucination_severity=1.0,
            )

        fact_consistency = self._check_fact_consistency(answer, documents)
        faithfulness = fact_consistency
        contradiction_rate = 1.0 - fact_consistency
        unsubstantiated = self._count_unsubstantiated_claims(answer, documents)

        hallucination_detected = fact_consistency < 0.6 or unsubstantiated > 3
        severity = max(0, (1 - fact_consistency) + (unsubstantiated * 0.1))

        return HallucinationMetrics(
            fact_consistency_score=fact_consistency,
            contradiction_rate=contradiction_rate,
            unsubstantiated_claims=unsubstantiated,
            hallucination_detected=hallucination_detected,
            hallucination_severity=severity,
            faithfulness_score=faithfulness,
        )

    def calculate_performance_metrics(
        self,
        total_time_ms: float,
        retrieval_time_ms: float,
        generation_time_ms: float,
        documents_count: int,
        search_iterations: int,
        token_count: int,
    ) -> PerformanceMetrics:
        """Calculate performance and efficiency metrics."""
        cost_estimate = token_count * 0.00001

        return PerformanceMetrics(
            total_latency_ms=total_time_ms,
            retrieval_latency_ms=retrieval_time_ms,
            generation_latency_ms=generation_time_ms,
            documents_retrieved=documents_count,
            search_iterations=search_iterations,
            token_usage=token_count,
            cost_estimate=cost_estimate,
        )

    def evaluate_query(
        self,
        query: str,
        answer: str,
        ground_truth: str,
        retrieved_docs: List[Union[Document, LangChainDocument]],
        gold_docs: List[Union[Document, LangChainDocument]],
        performance: PerformanceMetrics,
    ) -> dict:
        """Evaluate a single query and return all metrics."""
        answer_metrics = self.calculate_answer_metrics(answer, ground_truth, query)
        retrieval_metrics = self.calculate_retrieval_metrics(retrieved_docs, gold_docs)
        hallucination_metrics = self.calculate_hallucination_metrics(answer, retrieved_docs)

        return {
            "query": query,
            "answer_metrics": answer_metrics,
            "retrieval_metrics": retrieval_metrics,
            "hallucination_metrics": hallucination_metrics,
            "performance_metrics": performance,
            "overall_score": (
                answer_metrics.overall_score
                * (1 - hallucination_metrics.overall_hallucination_score)
                * performance.efficiency_score
            ),
        }

    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = " ".join(text.split())
        return text

    def _calculate_f1_score(self, predicted: str, ground_truth: str) -> float:
        """Calculate F1 score based on word overlap."""
        pred_words = set(predicted.split())
        gt_words = set(ground_truth.split())

        if not pred_words or not gt_words:
            return 0.0

        intersection = pred_words & gt_words
        precision = len(intersection) / len(pred_words)
        recall = len(intersection) / len(gt_words)

        if precision + recall == 0:
            return 0.0

        return 2 * (precision * recall) / (precision + recall)

    def _calculate_rouge_l_f1(self, predicted: str, ground_truth: str) -> float:
        """Calculate ROUGE-L F1 score."""
        pred_words = predicted.split()
        gt_words = ground_truth.split()

        if not pred_words or not gt_words:
            return 0.0

        lcs_length = self._lcs_length(pred_words, gt_words)
        recall = lcs_length / len(gt_words)
        precision = lcs_length / len(pred_words)

        if precision + recall == 0:
            return 0.0

        return 2 * (precision * recall) / (precision + recall)

    def _lcs_length(self, seq1: List[str], seq2: List[str]) -> int:
        """Calculate length of longest common subsequence."""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i - 1] == seq2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        return dp[m][n]

    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity (simplified)."""
        words1 = set(text1.split())
        words2 = set(text2.split())

        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union) if union else 0.0

    def _calculate_ndcg(
        self,
        retrieved: List[Document],
        relevant: List[Document],
        k: int,
    ) -> float:
        """Calculate NDCG@k."""
        if not retrieved or not relevant:
            return 0.0

        relevant_ids = set(
            doc.metadata.get("id", i)
            for i, doc in enumerate(relevant)
        )

        dcg = 0.0
        for i, doc in enumerate(retrieved[:k]):
            rid = doc.metadata.get("id", i)
            if rid in relevant_ids:
                rel_i = 1
                dcgi = 1 / np.log2(rel_i + 2)
                dcg += dcgi

        ideal_relevances = [1] * min(len(relevant_ids), k)
        idcg = sum(
            1 / np.log2(i + 2)
            for i in range(len(ideal_relevances))
        )

        return dcg / idcg if idcg > 0 else 0.0

    def _check_fact_consistency(
        self,
        answer: str,
        documents: List[Union[Document, LangChainDocument]],
    ) -> float:
        """Check fact consistency between answer and documents."""
        if not documents:
            return 0.0

        answer_lower = answer.lower()
        doc_texts = " ".join([doc.page_content for doc in documents]).lower()

        words_in_answer = set(answer_lower.split())
        words_in_docs = set(doc_texts.split())

        overlap = len(words_in_answer & words_in_docs)
        total_words = len(words_in_answer)

        return overlap / total_words if total_words > 0 else 0.0

    def _count_unsubstantiated_claims(
        self,
        answer: str,
        documents: List[Union[Document, LangChainDocument]],
    ) -> int:
        """Count claims in answer not supported by documents."""
        sentences = [s.strip() for s in re.split(r'[.!?]+', answer)]
        unsubstantiated = 0
        doc_text = " ".join([doc.page_content for doc in documents]).lower()

        for sentence in sentences:
            if sentence and not self._sentence_in_context(sentence, doc_text):
                unsubstantiated += 1

        return unsubstantiated

    def _sentence_in_context(self, sentence: str, context: str) -> bool:
        """Check if sentence is supported by context."""
        sentence_lower = sentence.lower().strip()
        words = [w for w in sentence_lower.split() if len(w) > 4]
        if not words:
            return False
        return any(word in context for word in words)

    def aggregate_metrics(
        self,
        query_results: List[dict],
    ) -> dict:
        """Aggregate metrics across multiple queries."""
        if not query_results:
            return {}

        f1_scores = [r["answer_metrics"].f1_score for r in query_results]
        precision_scores = [r["retrieval_metrics"].precision_at_k for r in query_results]
        recall_scores = [r["retrieval_metrics"].recall_at_k for r in query_results]
        hallucination_rates = [
            1 - r["hallucination_metrics"].fact_consistency_score
            for r in query_results
        ]
        latencies = [r["performance_metrics"].total_latency_ms for r in query_results]
        overall_scores = [r["overall_score"] for r in query_results]

        return {
            "count": len(query_results),
            "avg_f1_score": float(np.mean(f1_scores)),
            "std_f1_score": float(np.std(f1_scores)),
            "avg_precision_at_k": float(np.mean(precision_scores)),
            "avg_recall_at_k": float(np.mean(recall_scores)),
            "avg_hallucination_rate": float(np.mean(hallucination_rates)),
            "avg_latency_ms": float(np.mean(latencies)),
            "avg_overall_score": float(np.mean(overall_scores)),
            "max_overall_score": float(np.max(overall_scores)),
            "min_overall_score": float(np.min(overall_scores)),
        }
