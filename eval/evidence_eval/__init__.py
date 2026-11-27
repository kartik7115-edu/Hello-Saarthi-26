"""Evidence grounding and citation-quality evaluation for Saarthi (AI Architects, Sprint 6).

annotations.py    strict, versioned schema for human claim/evidence annotations and their provenance
records.py        recorded answers and their retrieval context (Sprint 5 recorded-result format)
resolution.py     citation resolution against the retrieval context; citation-to-claim association
metrics.py        claim outcomes and citation/grounding metrics (no weighted score)
evaluator.py      joins answers with annotations; report with findings and data-quality problems
run_evaluation.py CLI

Evaluates RECORDED answers only; it never generates an answer or runs a retriever. Imports only
public pieces of eval.answer_eval (citation parser, dataset loader) and eval.retrieval_eval; neither
imports this package. See README.md.
"""
