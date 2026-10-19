"""Scenario-based evals: deterministic tool-trace assertions plus an LLM judge.

The two halves do different jobs. The trace assertions catch wiring regressions cheaply and
without ambiguity - did the agent actually consult the customer database? The judge scores the
things no assertion can express - is the answer grounded, did it honour the constraint.
"""
