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

event_sig_InitialisedContract = web3.Web3.sha3(text='InitialisedContract(address,uint256)')
event_InitialisedContract = w3.eth.filter({'topics': [event_sig_InitialisedContract]})

""" Supply token-pool to creator """
balanceOf_creator_before = w3.eth.getBalance(creator_address)
print('balanceOf_creator before', balanceOf_creator_before)
init_event = w3.eth.getFilterChanges(event_InitialisedContract.filter_id)
print('init_event', init_event)

setter_initialise_tokens(w3, contract_instance, deployment_tx_hash, contract_address)

balanceOf_creator_after = w3.eth.getBalance(creator_address)
print('balanceOf_creator after', balanceOf_creator_after)
new_init_event = w3.eth.getFilterChanges(event_InitialisedContract.filter_id)
print('new_init_event', init_event)

print('get all Logs?', event_InitialisedContract.get)
print('break')


""" Make a promise"""
promiser = w3.eth.accounts[1]
promise = 1000                   #   for now both consuming and producing promise
print('balance before promise,', w3.eth.getBalance(promiser))
setter_promise_sell(w3, contract_instance, promiser, promise)
print('balance after promise,', w3.eth.getBalance(promiser))

print('balance of,', contract_instance.balanceOf(w3.eth.accounts[0]))

""" Mint a token """
producer = w3.eth.accounts[1]
value = 9
setter_mint(w3, contract_instance, producer, value)
new_mint_events = w3.eth.getFilterChanges(event_CreatedEnergy.filter_id)
print("mint event,", new_mint_events)

print('balance of,', contract_instance.balanceOf(w3.eth.accounts[1]))

#
# """ Burn a token in exchange of consumption"""
# consumer = w3.eth.accounts[1]
# value = 8
# setter_burn(w3, contract_instance, consumer, value)
# new_burn_events = w3.eth.getFilterChanges(event_RemovedEnergy.filter_id)
# print("burn event,", new_burn_events)
#
# print('done')
#
# # setter_mint(w3, contract_instance, deployment_tx_hash, minter)
#
#






