// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

contract FedAlphaGovernance {
    uint256 public constant MIN_STAKE = 100 ether;
    uint256 public constant SLASH_RATIO = 20;
    uint256 public constant REWARD_POOL = 10 ether;

    address public oracleAddress;

    struct Participant {
        address addr;
        uint256 stake;
        uint256 reputation;
        uint256 roundsParticipated;
        uint256 roundsValidated;
        bool isSlashed;
    }

    mapping(address => Participant) public participants;

    event ParticipantStaked(address indexed participant, uint256 amount);
    event ParticipantSlashed(address indexed participant, uint256 amount, string reason);
    event RewardDistributed(address indexed aggregator, uint256 amount);
    event ModelValidated(bytes32 modelHash, uint256 sharpeRatio, bool passed);

    modifier onlyOracle() {
        require(msg.sender == oracleAddress, "Oracle only");
        _;
    }

    constructor(address _oracleAddress) payable {
        oracleAddress = _oracleAddress;
    }

    function stake() external payable {
        require(msg.value >= MIN_STAKE, "Insufficient stake");
        Participant storage participant = participants[msg.sender];
        participant.addr = msg.sender;
        participant.stake += msg.value;
        if (participant.reputation == 0) {
            participant.reputation = 100;
        }
        emit ParticipantStaked(msg.sender, msg.value);
    }

    function recordParticipation(address participantAddress) external onlyOracle {
        participants[participantAddress].roundsParticipated += 1;
    }

    function slash(address participantAddress, string calldata reason) external onlyOracle {
        Participant storage participant = participants[participantAddress];
        uint256 slashAmount = participant.stake * SLASH_RATIO / 100;
        participant.stake -= slashAmount;
        participant.isSlashed = true;
        participant.reputation = participant.reputation > SLASH_RATIO
            ? participant.reputation - SLASH_RATIO
            : 0;
        emit ParticipantSlashed(participantAddress, slashAmount, reason);
    }

    function distributeReward(address aggregator, bytes32 modelHash, uint256 sharpeRatio) external onlyOracle {
        Participant storage participant = participants[aggregator];
        participant.roundsValidated += 1;
        participant.reputation = computeReputation(aggregator);
        require(address(this).balance >= REWARD_POOL, "Reward pool empty");
        payable(aggregator).transfer(REWARD_POOL);
        emit RewardDistributed(aggregator, REWARD_POOL);
        emit ModelValidated(modelHash, sharpeRatio, true);
    }

    function computeReputation(address participantAddress) public view returns (uint256) {
        Participant memory participant = participants[participantAddress];
        if (participant.roundsParticipated == 0) {
            return participant.reputation == 0 ? 100 : participant.reputation;
        }
        return participant.roundsValidated * 100 / participant.roundsParticipated;
    }

    receive() external payable {}
}
