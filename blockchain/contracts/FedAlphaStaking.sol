// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

contract FedAlphaStaking {
    uint256 public immutable minStake;
    address public owner;
    address public slashingManager;
    mapping(address => uint256) public stakes;

    event StakeDeposited(address indexed participant, uint256 amount, uint256 totalStake);
    event StakeWithdrawn(address indexed participant, uint256 amount, uint256 remainingStake);
    event ParticipantJoinedRound(address indexed participant, uint256 indexed roundId, uint256 stake);
    event SlashingManagerUpdated(address indexed slashingManager);
    event StakeSlashed(address indexed participant, uint256 amount, address indexed recipient, uint256 remainingStake);

    modifier onlyOwner() {
        require(msg.sender == owner, "Owner only");
        _;
    }

    modifier onlySlashingManager() {
        require(msg.sender == slashingManager, "Slashing manager only");
        _;
    }

    constructor(uint256 _minStake) {
        require(_minStake > 0, "Invalid min stake");
        minStake = _minStake;
        owner = msg.sender;
    }

    function setSlashingManager(address _slashingManager) external onlyOwner {
        require(_slashingManager != address(0), "Invalid slashing manager");
        slashingManager = _slashingManager;
        emit SlashingManagerUpdated(_slashingManager);
    }

    function deposit() external payable {
        require(msg.value > 0, "Stake required");
        stakes[msg.sender] += msg.value;
        emit StakeDeposited(msg.sender, msg.value, stakes[msg.sender]);
    }

    function withdraw(uint256 amount) external {
        require(amount > 0, "Amount required");
        require(stakes[msg.sender] >= amount, "Insufficient stake");

        stakes[msg.sender] -= amount;
        emit StakeWithdrawn(msg.sender, amount, stakes[msg.sender]);
        payable(msg.sender).transfer(amount);
    }

    function canParticipate(address participant) public view returns (bool) {
        return stakes[participant] >= minStake;
    }

    function joinRound(uint256 roundId) external {
        require(canParticipate(msg.sender), "Minimum stake required");
        emit ParticipantJoinedRound(msg.sender, roundId, stakes[msg.sender]);
    }

    function slash(address participant, uint256 amount, address recipient)
        external
        onlySlashingManager
        returns (uint256)
    {
        require(participant != address(0), "Invalid participant");
        require(recipient != address(0), "Invalid recipient");
        require(amount > 0, "Amount required");

        uint256 currentStake = stakes[participant];
        require(currentStake > 0, "No stake");
        uint256 slashAmount = amount > currentStake ? currentStake : amount;
        stakes[participant] = currentStake - slashAmount;
        emit StakeSlashed(participant, slashAmount, recipient, stakes[participant]);
        payable(recipient).transfer(slashAmount);
        return slashAmount;
    }
}
