const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("FedRegistry", function () {
  it("records model hashes per FL round and verifies by oracle only", async function () {
    const [owner, oracle, participantA, participantB] = await ethers.getSigners();
    const Registry = await ethers.getContractFactory("FedRegistry");
    const registry = await Registry.deploy(oracle.address);

    const hashA = ethers.keccak256(ethers.toUtf8Bytes("round-1-client-a"));
    const hashB = ethers.keccak256(ethers.toUtf8Bytes("round-1-client-b"));
    const aggregateHash = ethers.keccak256(ethers.toUtf8Bytes("round-1-aggregate"));

    await expect(registry.connect(participantA).submitModelHash(1, hashA))
      .to.emit(registry, "ModelHashSubmitted")
      .withArgs(1, participantA.address, hashA);
    await registry.connect(participantB).submitModelHash(1, hashB);

    expect(await registry.roundHashCount(1)).to.equal(2);
    expect(await registry.getRoundHashes(1)).to.deep.equal([hashA, hashB]);
    expect(await registry.submitterOf(1, hashA)).to.equal(participantA.address);

    await expect(registry.connect(participantA).verifyRound(1, aggregateHash)).to.be.revertedWith(
      "Oracle only"
    );
    await expect(registry.connect(oracle).verifyRound(1, aggregateHash))
      .to.emit(registry, "RoundVerified")
      .withArgs(1, aggregateHash);

    expect(await registry.roundVerified(1)).to.equal(true);
    expect(await registry.verifiedAggregateHash(1)).to.equal(aggregateHash);
    await expect(registry.connect(oracle).verifyRound(1, aggregateHash)).to.be.revertedWith(
      "Round already verified"
    );
  });

  it("rejects empty and duplicate hashes", async function () {
    const [owner, oracle, participant] = await ethers.getSigners();
    const Registry = await ethers.getContractFactory("FedRegistry");
    const registry = await Registry.deploy(oracle.address);
    const modelHash = ethers.keccak256(ethers.toUtf8Bytes("duplicate"));

    await expect(registry.connect(participant).submitModelHash(1, ethers.ZeroHash)).to.be.revertedWith(
      "Empty hash"
    );
    await registry.connect(participant).submitModelHash(1, modelHash);
    await expect(registry.connect(participant).submitModelHash(1, modelHash)).to.be.revertedWith(
      "Hash already submitted"
    );
    await expect(registry.connect(oracle).verifyRound(2, modelHash)).to.be.revertedWith(
      "No submissions"
    );
  });

  it("anchors IPFS checkpoint CID hashes without storing model payloads", async function () {
    const [owner, oracle, participant] = await ethers.getSigners();
    const Registry = await ethers.getContractFactory("FedRegistry");
    const registry = await Registry.deploy(oracle.address);
    const modelHash = ethers.keccak256(ethers.toUtf8Bytes("round-2-client-a"));
    const cidHash = ethers.keccak256(ethers.toUtf8Bytes("ipfs://bafy-round-2-client-a"));

    await expect(registry.connect(participant).submitModelCheckpoint(2, modelHash, cidHash))
      .to.emit(registry, "CheckpointAnchored")
      .withArgs(2, participant.address, modelHash, cidHash);

    expect(await registry.checkpointCidHash(2, modelHash)).to.equal(cidHash);
    await expect(
      registry.connect(owner).anchorCheckpoint(2, modelHash, cidHash)
    ).to.be.revertedWith("Submitter only");
  });
});

describe("FedAlphaStaking", function () {
  it("requires minimum stake before a participant joins a round", async function () {
    const [owner, participant] = await ethers.getSigners();
    const Staking = await ethers.getContractFactory("FedAlphaStaking");
    const staking = await Staking.deploy(ethers.parseEther("100"));

    await expect(staking.connect(participant).joinRound(1)).to.be.revertedWith(
      "Minimum stake required"
    );
    await expect(staking.connect(participant).deposit({ value: ethers.parseEther("100") }))
      .to.emit(staking, "StakeDeposited")
      .withArgs(participant.address, ethers.parseEther("100"), ethers.parseEther("100"));

    expect(await staking.canParticipate(participant.address)).to.equal(true);
    await expect(staking.connect(participant).joinRound(1))
      .to.emit(staking, "ParticipantJoinedRound")
      .withArgs(participant.address, 1, ethers.parseEther("100"));
  });

  it("lets participants withdraw and updates eligibility", async function () {
    const [owner, participant] = await ethers.getSigners();
    const Staking = await ethers.getContractFactory("FedAlphaStaking");
    const staking = await Staking.deploy(ethers.parseEther("100"));

    await staking.connect(participant).deposit({ value: ethers.parseEther("120") });
    await expect(() =>
      staking.connect(participant).withdraw(ethers.parseEther("30"))
    ).to.changeEtherBalances([participant, staking], [ethers.parseEther("30"), -ethers.parseEther("30")]);

    expect(await staking.stakes(participant.address)).to.equal(ethers.parseEther("90"));
    expect(await staking.canParticipate(participant.address)).to.equal(false);
    await expect(staking.connect(participant).withdraw(ethers.parseEther("91"))).to.be.revertedWith(
      "Insufficient stake"
    );
  });

  it("delegates slashing to a slashing manager", async function () {
    const [owner, oracle, treasury, participant] = await ethers.getSigners();
    const Staking = await ethers.getContractFactory("FedAlphaStaking");
    const staking = await Staking.deploy(ethers.parseEther("100"));
    const SlashingManager = await ethers.getContractFactory("SlashingManager");
    const manager = await SlashingManager.deploy(
      await staking.getAddress(),
      oracle.address,
      treasury.address,
      2500
    );
    await staking.setSlashingManager(await manager.getAddress());
    await staking.connect(participant).deposit({ value: ethers.parseEther("100") });

    const modelHash = ethers.keccak256(ethers.toUtf8Bytes("invalid-model"));
    await expect(() =>
      manager.connect(oracle).slashInvalidModel(3, participant.address, modelHash, false, "oracle failed")
    ).to.changeEtherBalances([treasury, staking], [ethers.parseEther("25"), -ethers.parseEther("25")]);

    expect(await staking.stakes(participant.address)).to.equal(ethers.parseEther("75"));
    await expect(
      manager.connect(participant).slashInvalidModel(3, participant.address, modelHash, false, "bad")
    ).to.be.revertedWith("Oracle only");
  });
});
