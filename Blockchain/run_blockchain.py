from Blockchain.SmartContract import*

test_environment_eth = object
Bazaar = classmethod

"""Compile smart contract from Solidity"""
contract_interface, w3 = compile_smart_contract()

# print(contract_interface)


"""Deploy smart contract in genesis transaction"""
w3, contract_instance, deployment_tx_hash, contract_address = deploy_SC(contract_interface, w3)

setters(w3, contract_instance, deployment_tx_hash, contract_address)