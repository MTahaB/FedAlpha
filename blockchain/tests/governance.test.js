const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("FedAlphaGovernance", function () {
  it("stakes and slashes participants", async function () {
    const [owner, oracle, participant] = await ethers.getSigners();
    const Governance = await ethers.getContractFactory("FedAlphaGovernance");
    const governance = await Governance.deploy(oracle.address, {
      value: ethers.parseEther("100")
    });

    await governance.connect(participant).stake({ value: ethers.parseEther("100") });
    expect((await governance.participants(participant.address)).stake).to.equal(ethers.parseEther("100"));

    await governance.connect(oracle).slash(participant.address, "byzantine update");
    expect((await governance.participants(participant.address)).stake).to.equal(ethers.parseEther("80"));
  });
});
