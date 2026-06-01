const hre = require("hardhat");

async function main() {
  const [deployer, oracle] = await hre.ethers.getSigners();

  const Governance = await hre.ethers.getContractFactory("FedAlphaGovernance");
  const governance = await Governance.deploy(oracle.address, {
    value: hre.ethers.parseEther("100")
  });
  await governance.waitForDeployment();

  const DAO = await hre.ethers.getContractFactory("FedAlphaDAO");
  const dao = await DAO.deploy([deployer.address], [100]);
  await dao.waitForDeployment();

  console.log({
    governance: await governance.getAddress(),
    dao: await dao.getAddress(),
    oracle: oracle.address
  });
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
