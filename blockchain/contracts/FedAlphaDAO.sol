// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

contract FedAlphaDAO {
    struct Proposal {
        string description;
        uint256 newMinSharpe;
        uint256 newMaxDrawdown;
        uint256 votesFor;
        uint256 votesAgainst;
        uint256 deadline;
        bool executed;
    }

    mapping(uint256 => Proposal) public proposals;
    mapping(address => uint256) public votingPower;
    uint256 public proposalCount;
    uint256 public minSharpe = 75;
    uint256 public maxDrawdown = 20;

    event Proposed(uint256 indexed proposalId, string description);
    event Voted(uint256 indexed proposalId, address indexed voter, bool support, uint256 weight);
    event Executed(uint256 indexed proposalId, uint256 minSharpe, uint256 maxDrawdown);

    constructor(address[] memory initialVoters, uint256[] memory powers) {
        require(initialVoters.length == powers.length, "Length mismatch");
        for (uint256 i = 0; i < initialVoters.length; i++) {
            votingPower[initialVoters[i]] = powers[i];
        }
    }

    function propose(
        string calldata description,
        uint256 newMinSharpe,
        uint256 newMaxDrawdown
    ) external returns (uint256) {
        require(votingPower[msg.sender] > 0, "No voting power");
        proposals[proposalCount] = Proposal({
            description: description,
            newMinSharpe: newMinSharpe,
            newMaxDrawdown: newMaxDrawdown,
            votesFor: 0,
            votesAgainst: 0,
            deadline: block.timestamp + 7 days,
            executed: false
        });
        emit Proposed(proposalCount, description);
        return proposalCount++;
    }

    function vote(uint256 proposalId, bool support) external {
        Proposal storage proposal = proposals[proposalId];
        require(block.timestamp < proposal.deadline, "Voting ended");
        uint256 weight = votingPower[msg.sender];
        require(weight > 0, "No voting power");
        if (support) {
            proposal.votesFor += weight;
        } else {
            proposal.votesAgainst += weight;
        }
        emit Voted(proposalId, msg.sender, support, weight);
    }

    function execute(uint256 proposalId) external {
        Proposal storage proposal = proposals[proposalId];
        require(block.timestamp >= proposal.deadline, "Voting active");
        require(!proposal.executed, "Already executed");
        require(proposal.votesFor > proposal.votesAgainst, "Proposal rejected");
        proposal.executed = true;
        minSharpe = proposal.newMinSharpe;
        maxDrawdown = proposal.newMaxDrawdown;
        emit Executed(proposalId, minSharpe, maxDrawdown);
    }
}
