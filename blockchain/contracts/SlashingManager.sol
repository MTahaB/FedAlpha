// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

interface IFedAlphaStaking {
    function stakes(address participant) external view returns (uint256);
    function slash(address participant, uint256 amount, address recipient) external returns (uint256);
}

contract SlashingManager {
    IFedAlphaStaking public immutable staking;
    address public oracle;
    address public treasury;
    uint256 public partialSlashBps;

    event InvalidModelSlashed(
        uint256 indexed roundId,
        address indexed participant,
        bytes32 indexed modelHash,
        uint256 amount,
        bool totalSlash,
        string reason
    );

    modifier onlyOracle() {
        require(msg.sender == oracle, "Oracle only");
        _;
    }

    constructor(address _staking, address _oracle, address _treasury, uint256 _partialSlashBps) {
        require(_staking != address(0), "Invalid staking");
        require(_oracle != address(0), "Invalid oracle");
        require(_treasury != address(0), "Invalid treasury");
        require(_partialSlashBps > 0 && _partialSlashBps <= 10_000, "Invalid slash bps");
        staking = IFedAlphaStaking(_staking);
        oracle = _oracle;
        treasury = _treasury;
        partialSlashBps = _partialSlashBps;
    }

    function slashInvalidModel(
        uint256 roundId,
        address participant,
        bytes32 modelHash,
        bool totalSlash,
        string calldata reason
    ) external onlyOracle returns (uint256) {
        require(modelHash != bytes32(0), "Empty hash");
        uint256 currentStake = staking.stakes(participant);
        require(currentStake > 0, "No stake");
        uint256 amount = totalSlash ? currentStake : currentStake * partialSlashBps / 10_000;
        uint256 slashed = staking.slash(participant, amount, treasury);
        emit InvalidModelSlashed(roundId, participant, modelHash, slashed, totalSlash, reason);
        return slashed;
    }
}
