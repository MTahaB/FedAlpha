const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("MultiSigOracle", function () {
  it("finalizes validation after threshold and verifies registry on-chain", async function () {
    const [owner, validatorA, validatorB, validatorC, participant] = await ethers.getSigners();
    const Oracle = await ethers.getContractFactory("MultiSigOracle");
    const oracle = await Oracle.deploy([validatorA.address, validatorB.address, validatorC.address], 2);
    const Registry = await ethers.getContractFactory("FedRegistry");
    const registry = await Registry.deploy(await oracle.getAddress());

    const clientHash = ethers.keccak256(ethers.toUtf8Bytes("client-checkpoint"));
    const aggregateHash = ethers.keccak256(ethers.toUtf8Bytes("aggregate-checkpoint"));
    await registry.connect(participant).submitModelHash(7, clientHash);

    await oracle.connect(validatorA).submitValidation(7, aggregateHash, true, 91);
    expect((await oracle.validations(7)).finalized).to.equal(false);
    await expect(oracle.connect(validatorB).submitValidation(7, aggregateHash, true, 93))
      .to.emit(oracle, "OracleRoundFinalized")
      .withArgs(7, aggregateHash, true);

    await oracle.connect(validatorA).verifyRegistryRound(await registry.getAddress(), 7);
    expect(await registry.roundVerified(7)).to.equal(true);
    expect(await registry.verifiedAggregateHash(7)).to.equal(aggregateHash);
  });

  it("rejects duplicate votes and hash mismatches", async function () {
    const [owner, validatorA, validatorB] = await ethers.getSigners();
    const Oracle = await ethers.getContractFactory("MultiSigOracle");
    const oracle = await Oracle.deploy([validatorA.address, validatorB.address], 2);
    const hashA = ethers.keccak256(ethers.toUtf8Bytes("hash-a"));
    const hashB = ethers.keccak256(ethers.toUtf8Bytes("hash-b"));

    await oracle.connect(validatorA).submitValidation(1, hashA, true, 80);
    await expect(oracle.connect(validatorA).submitValidation(1, hashA, true, 80)).to.be.revertedWith(
      "Already voted"
    );
    await expect(oracle.connect(validatorB).submitValidation(1, hashB, true, 80)).to.be.revertedWith(
      "Hash mismatch"
    );
  });

  it("does not verify rejected rounds", async function () {
    const [owner, validatorA, validatorB, participant] = await ethers.getSigners();
    const Oracle = await ethers.getContractFactory("MultiSigOracle");
    const oracle = await Oracle.deploy([validatorA.address, validatorB.address], 2);
    const Registry = await ethers.getContractFactory("FedRegistry");
    const registry = await Registry.deploy(await oracle.getAddress());
    const clientHash = ethers.keccak256(ethers.toUtf8Bytes("client"));
    const aggregateHash = ethers.keccak256(ethers.toUtf8Bytes("aggregate"));

    await registry.connect(participant).submitModelHash(8, clientHash);
    await oracle.connect(validatorA).submitValidation(8, aggregateHash, false, 12);
    await oracle.connect(validatorB).submitValidation(8, aggregateHash, false, 10);

    await expect(oracle.connect(validatorA).verifyRegistryRound(await registry.getAddress(), 8)).to.be.revertedWith(
      "Round rejected"
    );
  });
});
