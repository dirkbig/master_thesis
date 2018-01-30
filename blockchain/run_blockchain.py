from blockchain.smartcontract import*
import web3

test_environment_eth = object
Bazaar = classmethod

""" Compile smart contract from Solidity """
contract_interface, w3 = compile_smart_contract()
creator_address =  w3.eth.accounts[0]

""" Deploy smart contract in genesis transaction """
w3, contract_instance, deployment_tx_hash, contract_address = deploy_SC(contract_interface, w3, creator_address)

"""create event filters"""
event_sig_Transfer = web3.Web3.sha3(text='Transfer(address,address,uint256)')
event_Transfer = w3.eth.filter({'topics': [event_sig_Transfer]})

event_sig_Donate = web3.Web3.sha3(text='Donate(address,address,uint256)')
event_Donate = w3.eth.filter({'topics': [event_sig_Donate]})

event_sig_CreatedEnergy = web3.Web3.sha3(text='CreatedEnergy(address,address,uint256)')
event_CreatedEnergy = w3.eth.filter({'topics': [event_sig_CreatedEnergy]})

event_sig_RemovedEnergy = web3.Web3.sha3(text='RemovedEnergy(address,address,uint256)')
event_RemovedEnergy = w3.eth.filter({'topics': [event_sig_RemovedEnergy]})

event_sig_InitialisedContract = web3.Web3.sha3(text='InitialisedContract()')
event_InitialisedContract = w3.eth.filter({'topics': [event_sig_InitialisedContract]})

""" Supply token-pool to creator """
balanceOf_creator_before = w3.eth.getBalance(creator_address)
print('balanceOf_creator before', balanceOf_creator_before)
setter_initialise_tokens(w3, contract_instance, deployment_tx_hash, contract_address)

balanceOf_creator = w3.eth.getBalance(creator_address)
print('balanceOf_creator after', balanceOf_creator)
print(balanceOf_creator - balanceOf_creator_before)


transfer_events = w3.eth.getFilterChanges(event_InitialisedContract.filter_id)
print('transfer_events', transfer_events)


setter_initialise_tokens(w3, contract_instance, deployment_tx_hash, contract_address)

new_transfer_events = w3.eth.getFilterChanges(event_InitialisedContract.filter_id)



print('transfer_events', transfer_events)
print('new_transfer_events', new_transfer_events)

