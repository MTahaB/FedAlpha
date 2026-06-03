// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

interface IFedRegistry {
    function verifyRound(uint256 roundId, bytes32 aggregateModelHash) external;
}

contract MultiSigOracle {
    mapping(address => bool) public isValidator;
    address[] public validators;
    uint256 public immutable threshold;

    struct RoundValidation {
        bytes32 aggregateModelHash;
        uint256 approvals;
        uint256 rejections;
        bool finalized;
        bool passed;
        uint256 validationScore;
    }

    mapping(uint256 => RoundValidation) public validations;
    mapping(uint256 => mapping(address => bool)) public hasVoted;

    event OracleVoteSubmitted(
        uint256 indexed roundId,
        address indexed validator,
        bytes32 indexed aggregateModelHash,
        bool passed,
        uint256 validationScore
    );
    event OracleRoundFinalized(uint256 indexed roundId, bytes32 indexed aggregateModelHash, bool passed);
    event RegistryRoundVerified(address indexed registry, uint256 indexed roundId, bytes32 aggregateModelHash);

    modifier onlyValidator() {
        require(isValidator[msg.sender], "Validator only");
        _;
    }

    constructor(address[] memory _validators, uint256 _threshold) {
        require(_validators.length > 0, "Validators required");
        require(_threshold > 0 && _threshold <= _validators.length, "Invalid threshold");
        threshold = _threshold;
        for (uint256 i = 0; i < _validators.length; i++) {
            address validator = _validators[i];
            require(validator != address(0), "Invalid validator");
            require(!isValidator[validator], "Duplicate validator");
            isValidator[validator] = true;
            validators.push(validator);
        }
    }

    function submitValidation(
        uint256 roundId,
        bytes32 aggregateModelHash,
        bool passed,
        uint256 validationScore
    ) external onlyValidator {
        require(aggregateModelHash != bytes32(0), "Empty aggregate hash");
        require(validationScore <= 100, "Invalid score");
        require(!hasVoted[roundId][msg.sender], "Already voted");

        RoundValidation storage validation = validations[roundId];
        if (validation.aggregateModelHash == bytes32(0)) {
            validation.aggregateModelHash = aggregateModelHash;
        } else {
            require(validation.aggregateModelHash == aggregateModelHash, "Hash mismatch");
        }

        hasVoted[roundId][msg.sender] = true;
        if (passed) {
            validation.approvals += 1;
        } else {
            validation.rejections += 1;
        }
        validation.validationScore = validationScore;
        emit OracleVoteSubmitted(roundId, msg.sender, aggregateModelHash, passed, validationScore);

        if (!validation.finalized && (validation.approvals >= threshold || validation.rejections >= threshold)) {
            validation.finalized = true;
            validation.passed = validation.approvals >= threshold;
            emit OracleRoundFinalized(roundId, aggregateModelHash, validation.passed);
        }
    }

    function verifyRegistryRound(address registry, uint256 roundId) external onlyValidator {
        require(registry != address(0), "Invalid registry");
        RoundValidation memory validation = validations[roundId];
        require(validation.finalized, "Round not finalized");
        require(validation.passed, "Round rejected");
        IFedRegistry(registry).verifyRound(roundId, validation.aggregateModelHash);
        emit RegistryRoundVerified(registry, roundId, validation.aggregateModelHash);
    }

    function validatorCount() external view returns (uint256) {
        return validators.length;
    }
}
