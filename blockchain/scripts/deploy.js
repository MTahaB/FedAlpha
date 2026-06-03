const hre = require("hardhat");

async function main() {
  const [deployer, oracleA, oracleB, oracleC, treasury] = await hre.ethers.getSigners();

  const Governance = await hre.ethers.getContractFactory("FedAlphaGovernance");
  const governance = await Governance.deploy(oracleA.address, {
    value: hre.ethers.parseEther("100")
  });
  await governance.waitForDeployment();

  const DAO = await hre.ethers.getContractFactory("FedAlphaDAO");
  const dao = await DAO.deploy([deployer.address], [100]);
  await dao.waitForDeployment();

  const MultiSigOracle = await hre.ethers.getContractFactory("MultiSigOracle");
  const multiSigOracle = await MultiSigOracle.deploy(
    [oracleA.address, oracleB.address, oracleC.address],
    2
  );
  await multiSigOracle.waitForDeployment();

  const Staking = await hre.ethers.getContractFactory("FedAlphaStaking");
  const staking = await Staking.deploy(hre.ethers.parseEther("100"));
  await staking.waitForDeployment();

  const Registry = await hre.ethers.getContractFactory("FedRegistry");
  const registry = await Registry.deploy(await multiSigOracle.getAddress());
  await registry.waitForDeployment();

  const SlashingManager = await hre.ethers.getContractFactory("SlashingManager");
  const slashingManager = await SlashingManager.deploy(
    await staking.getAddress(),
    await multiSigOracle.getAddress(),
    treasury.address,
    2500
  );
  await slashingManager.waitForDeployment();
  await staking.setSlashingManager(await slashingManager.getAddress());

  console.log({
    governance: await governance.getAddress(),
    dao: await dao.getAddress(),
    multiSigOracle: await multiSigOracle.getAddress(),
    registry: await registry.getAddress(),
    staking: await staking.getAddress(),
    slashingManager: await slashingManager.getAddress(),
    oracleValidators: [oracleA.address, oracleB.address, oracleC.address],
    treasury: treasury.address
  });
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
