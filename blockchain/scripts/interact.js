const hre = require("hardhat");

async function main() {
  const address = process.env.GOVERNANCE_ADDRESS;
  if (!address) {
    throw new Error("Set GOVERNANCE_ADDRESS");
  }
  const governance = await hre.ethers.getContractAt("FedAlphaGovernance", address);
  console.log({ minStake: (await governance.MIN_STAKE()).toString() });
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
