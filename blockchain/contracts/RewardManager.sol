// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

contract RewardManager {
    event RewardPaid(address indexed recipient, uint256 amount, bytes32 modelHash);

    function payReward(address payable recipient, bytes32 modelHash) external payable {
        require(msg.value > 0, "Reward required");
        recipient.transfer(msg.value);
        emit RewardPaid(recipient, msg.value, modelHash);
    }
}
