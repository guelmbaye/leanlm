"""LeanLM(tm) -- Offline LLM Optimization Layer for Commodity Laptops.

LeanLM is not a language model. It is a deterministic, resource-aware inference
runtime that makes an existing GGUF model usable on a ~8 GB RAM laptop.

Architecture: Capability-Oriented Architecture (COA)
Contract:     Deterministic Inference Contract (DIC)
Runtime:      llama.cpp / GGUF
"""

__version__ = "1.0.0"
__product__ = "LeanLM"
PIPELINE_VERSION = "1.0.0"
CONTRACT_VERSION = "1.0.0"
