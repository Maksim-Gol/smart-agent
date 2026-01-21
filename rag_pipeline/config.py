"""Configuration for the RAG pipeline."""

# Qdrant settings
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "smart_contract_vulnerabilities"

# OpenAI settings
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
LLM_MODEL = "gpt-4o-mini"

# Dataset path
DATASET_PATH = "../smartbugs-curated"
VULN_JSON_PATH = f"{DATASET_PATH}/vulnerabilities.json"

# Vulnerability descriptions
VULNERABILITY_DESCRIPTIONS = {
    "reentrancy": "Reentrant function calls make a contract behave unexpectedly. An attacker can recursively call a function before the first execution completes, draining funds.",
    "access_control": "Failure to properly restrict access to sensitive functions. Missing modifiers, incorrect use of tx.origin, or unprotected critical functions.",
    "arithmetic": "Integer overflow or underflow vulnerabilities. Calculations that exceed the maximum or go below zero without proper checks.",
    "unchecked_low_level_calls": "Low-level calls (call, delegatecall, send) that fail silently. Return values are not checked, leading to unexpected behavior.",
    "denial_of_service": "Contract can be made unusable. Unbounded loops, gas limit issues, or external call failures blocking execution.",
    "bad_randomness": "Predictable random number generation. Using block variables (blockhash, timestamp) that miners can manipulate.",
    "front_running": "Transaction ordering exploitation. Attackers can see pending transactions and front-run them for profit.",
    "time_manipulation": "Reliance on block.timestamp which miners can manipulate within bounds.",
    "short_addresses": "EVM accepts incorrectly padded arguments, leading to unexpected token transfers.",
    "other": "Other vulnerability types not classified in DASP taxonomy."
}

