// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

contract FedRegistry {
    address public oracleAddress;

    mapping(uint256 => bytes32[]) private roundHashes;
    mapping(uint256 => mapping(bytes32 => address)) public submitterOf;
    mapping(uint256 => mapping(bytes32 => bytes32)) public checkpointCidHash;
    mapping(uint256 => bool) public roundVerified;
    mapping(uint256 => bytes32) public verifiedAggregateHash;

    event ModelHashSubmitted(
        uint256 indexed roundId,
        address indexed participant,
        bytes32 indexed modelHash
    );
    event CheckpointAnchored(
        uint256 indexed roundId,
        address indexed participant,
        bytes32 indexed modelHash,
        bytes32 cidHash
    );
    event RoundVerified(uint256 indexed roundId, bytes32 indexed aggregateModelHash);

    modifier onlyOracle() {
        require(msg.sender == oracleAddress, "Oracle only");
        _;
    }

    constructor(address _oracleAddress) {
        require(_oracleAddress != address(0), "Invalid oracle");
        oracleAddress = _oracleAddress;
    }

    function _submitModelHash(uint256 roundId, bytes32 modelHash, address participant) internal returns (uint256) {
        require(modelHash != bytes32(0), "Empty hash");
        require(submitterOf[roundId][modelHash] == address(0), "Hash already submitted");

        submitterOf[roundId][modelHash] = participant;
        roundHashes[roundId].push(modelHash);
        emit ModelHashSubmitted(roundId, participant, modelHash);
        return roundHashes[roundId].length - 1;
    }

    function submitModelHash(uint256 roundId, bytes32 modelHash) external returns (uint256) {
        return _submitModelHash(roundId, modelHash, msg.sender);
    }

    function submitModelCheckpoint(uint256 roundId, bytes32 modelHash, bytes32 cidHash)
        external
        returns (uint256)
    {
        require(cidHash != bytes32(0), "Empty CID hash");
        uint256 index = _submitModelHash(roundId, modelHash, msg.sender);
        checkpointCidHash[roundId][modelHash] = cidHash;
        emit CheckpointAnchored(roundId, msg.sender, modelHash, cidHash);
        return index;
    }

    function anchorCheckpoint(uint256 roundId, bytes32 modelHash, bytes32 cidHash) external {
        require(cidHash != bytes32(0), "Empty CID hash");
        require(submitterOf[roundId][modelHash] == msg.sender, "Submitter only");
        checkpointCidHash[roundId][modelHash] = cidHash;
        emit CheckpointAnchored(roundId, msg.sender, modelHash, cidHash);
    }

    function verifyRound(uint256 roundId, bytes32 aggregateModelHash) external onlyOracle {
        require(!roundVerified[roundId], "Round already verified");
        require(roundHashes[roundId].length > 0, "No submissions");
        require(aggregateModelHash != bytes32(0), "Empty aggregate hash");

        roundVerified[roundId] = true;
        verifiedAggregateHash[roundId] = aggregateModelHash;
        emit RoundVerified(roundId, aggregateModelHash);
    }

    function getRoundHashes(uint256 roundId) external view returns (bytes32[] memory) {
        return roundHashes[roundId];
    }

    function roundHashCount(uint256 roundId) external view returns (uint256) {
        return roundHashes[roundId].length;
    }
}
