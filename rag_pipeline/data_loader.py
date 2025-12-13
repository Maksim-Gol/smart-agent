"""Load and prepare vulnerability data for embedding."""

import json
from pathlib import Path

from config import VULN_JSON_PATH, DATASET_PATH, VULNERABILITY_DESCRIPTIONS


def load_vulnerabilities() -> list[dict]:
    """Load all vulnerabilities from the dataset."""
    with open(VULN_JSON_PATH, "r") as f:
        contracts = json.load(f)
    
    vulnerabilities = []
    
    for contract in contracts:
        file_path = Path(DATASET_PATH) / contract["path"]
        
        if not file_path.exists():
            continue
        
        code = file_path.read_text()
        lines = code.split("\n")
        
        for vuln in contract.get("vulnerabilities", []):
            category = vuln["category"]
            vuln_lines = vuln.get("lines", [])
            
            # Extract vulnerable code snippet (with context)
            snippet = extract_snippet(lines, vuln_lines, context=3)
            
            vulnerabilities.append({
                "contract_name": contract["name"],
                "file_path": contract["path"],
                "category": category,
                "vulnerable_lines": vuln_lines,
                "code_snippet": snippet,
                "full_code": code,
                "description": VULNERABILITY_DESCRIPTIONS.get(category, "Unknown vulnerability type")
            })
    
    return vulnerabilities


def extract_snippet(lines: list[str], vuln_lines: list[int], context: int = 3) -> str:
    """Extract code snippet around vulnerable lines."""
    if not vuln_lines:
        return "\n".join(lines[:20])  # First 20 lines if no specific lines
    
    min_line = max(0, min(vuln_lines) - context - 1)
    max_line = min(len(lines), max(vuln_lines) + context)
    
    return "\n".join(lines[min_line:max_line])


def create_embedding_text(vuln: dict) -> str:
    """
    Create text for embedding.
    
    This combines category, description, and code for rich semantic+syntactic representation.
    """
    return f"""Vulnerability Type: {vuln['category']}

Description: {vuln['description']}

Vulnerable Code:
```solidity
{vuln['code_snippet']}
```

Contract: {vuln['contract_name']}
Vulnerable Lines: {vuln['vulnerable_lines']}"""

