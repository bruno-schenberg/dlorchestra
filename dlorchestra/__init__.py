"""dlorchestra: DL experiment orchestration.

Orchestration only — does not ship model architectures or task-specific
training loops. Provides a config-driven experiment runner, a model registry
pattern, k-fold CV with stratification, metrics/plotting/artifact management,
AMP training with early stopping, and cross-model result analysis.
"""
