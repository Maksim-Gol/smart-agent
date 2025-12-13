"""RAG Agent for finding vulnerabilities in smart contracts."""

from openai import OpenAI

from config import LLM_MODEL
from embedder import Embedder
from vector_store import VectorStore


class VulnerabilityRAGAgent:
    """RAG Agent that finds vulnerabilities in smart contracts."""
    
    def __init__(self):
        self.embedder = Embedder()
        self.store = VectorStore()
        self.llm = OpenAI()
    
    def analyze_contract(self, contract_code: str, top_k: int = 5) -> str:
        """
        Analyze a smart contract for vulnerabilities.
        
        1. Embed the contract code
        2. Retrieve similar vulnerability patterns
        3. Use LLM to analyze and explain findings
        """
        # Step 1: Embed the query (contract code)
        query_embedding = self.embedder.embed(contract_code)
        
        # Step 2: Retrieve similar vulnerability patterns
        results = self.store.search(query_embedding, top_k=top_k)
        
        # Step 3: Build context from retrieved vulnerabilities
        context = self._build_context(results)
        
        # Step 4: Query LLM with RAG context
        analysis = self._query_llm(contract_code, context)
        
        return analysis
    
    def _build_context(self, results: list[dict]) -> str:
        """Build context string from search results."""
        context_parts = []
        
        for i, result in enumerate(results, 1):
            payload = result["payload"]
            context_parts.append(f"""
--- Similar Vulnerability #{i} (Similarity: {result['score']:.3f}) ---
Category: {payload['category']}
Description: {payload['description']}
Example from {payload['contract_name']}:
```solidity
{payload['code_snippet']}
```
""")
        
        return "\n".join(context_parts)
    
    def _query_llm(self, contract_code: str, context: str) -> str:
        """Query LLM with contract and vulnerability context."""
        prompt = f"""You are a smart contract security auditor. Analyze the following Solidity contract for vulnerabilities.

I've retrieved similar vulnerability patterns from a database of known vulnerabilities. Use these as reference:

{context}

Now analyze this contract:

```solidity
{contract_code}
```

Provide:
1. List of potential vulnerabilities found (with specific line references if possible)
2. Severity (Critical/High/Medium/Low/Informational)
3. Explanation of each vulnerability
4. Recommended fixes

Be specific and reference the similar patterns when applicable."""

        response = self.llm.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        
        return response.choices[0].message.content


def main():
    """Example usage."""
    # Sample vulnerable contract
    sample_contract = """
pragma solidity ^0.4.24;

contract VulnerableBank {
    mapping(address => uint256) public balances;
    
    function deposit() public payable {
        balances[msg.sender] += msg.value;
    }
    
    function withdraw(uint256 amount) public {
        require(balances[msg.sender] >= amount);
        
        // Vulnerable: external call before state update
        (bool success, ) = msg.sender.call.value(amount)("");
        require(success);
        
        balances[msg.sender] -= amount;
    }
    
    function getBalance() public view returns (uint256) {
        return balances[msg.sender];
    }
}
"""
    
    agent = VulnerabilityRAGAgent()
    print("Analyzing contract for vulnerabilities...\n")
    analysis = agent.analyze_contract(sample_contract)
    print(analysis)


if __name__ == "__main__":
    main()

