// SPDX-License-Identifier: MIT
pragma solidity ^0.4.24;

/**
 * Sample vulnerable contract for testing the RAG agent.
 * Contains multiple intentional vulnerabilities.
 */
contract VulnerableBank {
    mapping(address => uint256) public balances;
    address public owner;
    
    // Vulnerability 1: No constructor visibility (access_control)
    function VulnerableBank() public {
        owner = msg.sender;
    }
    
    function deposit() public payable {
        // Vulnerability 2: Integer overflow (arithmetic)
        balances[msg.sender] += msg.value;
    }
    
    function withdraw(uint256 amount) public {
        require(balances[msg.sender] >= amount);
        
        // Vulnerability 3: Reentrancy - external call before state update
        (bool success, ) = msg.sender.call.value(amount)("");
        require(success);
        
        balances[msg.sender] -= amount;
    }
    
    function transfer(address to, uint256 amount) public {
        // Vulnerability 4: Unchecked arithmetic
        require(balances[msg.sender] >= amount);
        balances[msg.sender] -= amount;
        balances[to] += amount;
    }
    
    // Vulnerability 5: tx.origin for authorization (access_control)
    function withdrawAll() public {
        require(tx.origin == owner);
        msg.sender.transfer(address(this).balance);
    }
    
    // Vulnerability 6: Unchecked low-level call
    function sendEther(address to, uint256 amount) public {
        to.call.value(amount)("");
    }
    
    function getBalance() public view returns (uint256) {
        return balances[msg.sender];
    }
}

