import json
import web3
import os
import sys
import time
from web3 import Web3, HTTPProvider, TestRPCProvider
from solc import compile_source
from web3.contract import ConciseContract

global contract_kind

# contract_kind = 'classic'
contract_kind = 'concise'

def compile_smart_contract():
    """ SOLIDITY CODE COMPILER,
        see https://github.com/pipermerriam/web3.py """

    contract_source_code = """ 
/*
     * A token contract where the tokens are minted every time new energy is created and destroyed at the other side.
     * Agent can approve its smart meter to use its tokens. The smart Meter can use those tokens on behalf its of owner.
     * A special donate function so people give back to the system if they want.
     */

    pragma solidity ^0.4.16;
    
    interface energyToken { function receiveApproval(address _from, uint256 _value, address _token, bytes _extraData) public; }
    
    contract HouseholdToken {
        // Public variables of the token
        uint256 public class;
        uint8 public decimals = 100;
        uint256 public totalSupply = 1000000;
        uint256 public startCapital = 1000000;
        uint256 public promise;
        // Private account of the contract creator. to be used as a base account
        address supplier;        
        
        
        // This creates an array with all balances
        // The actual balance
        mapping (address => uint256) public balanceOf;
        
        // mapping of promises of agents per step (needs to be updated each round)
        mapping (address => uint256) public promiseOfsell;
        mapping (address => uint256) public promiseOfbuy;
        mapping (address => uint256) public classificationOf;

        event Transfer(address indexed from, address indexed to, uint256 value);
        event Donate(address indexed from, uint256 value);
        event CreatedEnergy(address indexed from, uint256 value, uint256 balance);
        event RemovedEnergy(address indexed from, uint256 value);
        event InitialisedContract(address indexed from, uint256 value);
        event Test(uint256);
        
        
        //Initialisation of tokens
         
        function MyToken() public returns(uint256){
            // add require 1-time-give-away check
            supplier = msg.sender;                              // Storing the contract creator's address
            balanceOf[msg.sender] = totalSupply;                // Give the creator all initial tokens
            InitialisedContract(msg.sender, totalSupply);
            Test(balanceOf[msg.sender]);
            return totalSupply;

        }
                
        function giveStartingMoney() external {
            // add require 1-time-give-away check
            balanceOf[msg.sender] = startCapital; 
            InitialisedContract(msg.sender, totalSupply);
        }
        
        
        //Return functions
        function getBalance(address _user) constant returns (uint256){
            return balanceOf[_user];
            }
         
        function returnPromiseOfsell(address _account) constant returns (uint256){
            return promiseOfsell[_account];
            }
            
        function returnPromiseOfbuy(address _account) constant returns (uint256){
            return promiseOfbuy[_account];
            }


        //Internal transfer
        function _transfer(address _from, address _to, uint _value) private {
            require(_to != 0x0);                                         // Prevent transfer to 0x0 address. Use burn() instead
            require(balanceOf[_from] >= _value);                         // Check if the sender has enough
            require(balanceOf[_to] + _value >= balanceOf[_to]);          // Check for overflows
            //if (_class == 1)
            //     promise = promiseOfsell[_to] + 1;                       // Check whether agents holds its promise of selling
            //if (_class == 2)
            //    promise = promiseOfbuy[_from] + 1;                       // Check whether agents holds its promise of buying
            // require(_value <= promise);                                  // Check whether agents holds its promise of buying
            balanceOf[_from] -= _value;                                  // Subtract from the sender
            balanceOf[_to] += _value;                                    // Add the same to the recipient
            Transfer(_from, _to, _value);                                // Send Event
        }
    
    
        //
        function generatedEnergy(address _from, uint256 _value) public {
            require(_from != 0x0);                                            // Prevent transfer to 0x0 address. Use burn() instead
            require(balanceOf[msg.sender] >= _value);                         // Check if the sender has enough
            require(promiseOfsell[msg.sender] >= 0);
            balanceOf[supplier] -= _value;                                  // Subtract from the sender
            balanceOf[msg.sender] += _value;  
            CreatedEnergy(msg.sender, _value, balanceOf[msg.sender]);                             // Send Event
        }
    
         
        function consumedEnergy(address _from, uint256 _value) public {
            require(msg.sender != 0x0);                                            // Prevent transfer to 0x0 address. Use burn() instead
            require(balanceOf[msg.sender] >= _value);                               // Check if the sender has enough
            require(promiseOfbuy[msg.sender] >= 0);
            balanceOf[msg.sender] -= _value;  
            balanceOf[supplier] += _value;  
            RemovedEnergy(msg.sender, _value);                                           // Send Event
        }

        
        //make a promise on selling
        function makePromiseOfsell(address _promiser, uint256 _value) public returns (bool success) {
            promiseOfsell[_promiser] = _value;
            promiseOfbuy[_promiser] = 0;
            // classificationOf[_promiser] = 1;
            return true;
        }
        
        
        // make a promise on buying
        function makePromiseOfbuy(address _promiser, uint256 _value) public returns (bool success) {
            promiseOfbuy[_promiser] = _value;
            promiseOfsell[_promiser] = 0;
            // classificationOf[_promiser] = 2;
            return true;
        }
        
  
    }

    """

    compiled_sol = compile_source(contract_source_code) # Compiled source code
    contract_interface = compiled_sol['<stdin>:HouseholdToken']
    w3 = Web3(TestRPCProvider())

    return contract_interface, w3


def deploy_SC(contract_interface, w3, creator_address):
    """ Deploy smart_contract: : The Bazaar is opened"""
    """start virtual environment """
    # os.system('source ~/.venv-py3/bin/activate')
    # if hasattr(sys, 'real_prefix'):
    #     print('Virtual environment activated')
    contract_classic = w3.eth.contract(contract_interface['abi'], bytecode=contract_interface['bin'])
    tx_hash = contract_classic.deploy(transaction={'from': creator_address})
    tx_receipt = w3.eth.getTransactionReceipt(tx_hash)
    contract_address = tx_receipt['contractAddress']
    contract_instance  = contract_classic

    if contract_kind == 'concise':

        contract_concise = w3.eth.contract(contract_interface['abi'], contract_address,
                                            ContractFactoryClass=ConciseContract)
        contract_instance = contract_concise

    # # event_sig_Transfer = web3.Web3.sha3(text='Transfer(address,address,uint256)')
    # # event_Transfer = w3.eth.filter({'topics': [event_sig_Transfer]})
    #
    # # event_sig_Donate = web3.Web3.sha3(text='Donate(address,address,uint256)')
    # # event_Donate = w3.eth.filter({'topics': [event_sig_Donate]})
    #
    # event_sig_CreatedEnergy = web3.Web3.sha3(text='CreatedEnergy(address, uint256, uint256)')
    # event_CreatedEnergy = w3.eth.filter({'topics': [event_sig_CreatedEnergy]})
    #
    # # event_sig_RemovedEnergy = web3.Web3.sha3(text='RemovedEnergy(address,address,uint256)')
    # # event_RemovedEnergy = w3.eth.filter({'topics': [event_sig_RemovedEnergy]})
    #
    # event_sig_InitialisedContract = web3.Web3.sha3(text='InitialisedContract(address,uint256)')
    # event_InitialisedContract = w3.eth.filter({'topics': [event_sig_InitialisedContract]})

    return w3, contract_instance, tx_receipt, contract_address, event_CreatedEnergy, event_InitialisedContract

def setter_initialise_tokens(w3, contract_instance, deployment_tx_hash, creator_address, event_InitialisedContract):




    if contract_kind == 'classic':
        contract_instance.transact({'from': creator_address, 'to': creator_address}).MyToken()

    if contract_kind == 'concise':
        """ All transactions are simple calls"""
        contract_instance.MyToken(transact={'from': w3.eth.accounts[0]})
        print('Contract value: {}'.format(contract_instance.getBalance(w3.eth.accounts[0])))

    return



def setter_initialise_tokens2(w3, contract_instance, deployment_tx_hash, initiator):
    #
    # constructor_supply_hash = contract_instance.transact({'from': initiator, 'to': initiator}).MyToken2()
    # constructor_supply_receipt = w3.eth.getTransactionReceipt(constructor_supply_hash)
    # while constructor_supply_receipt is None:
    #     time.sleep(1)
    #     constructor_supply_receipt = w3.eth.getTransactionReceipt(constructor_supply_hash)


    if contract_kind == 'classic':
        contract_instance.transact({'from': initiator, 'to': initiator}).giveStartingMoney()

    elif contract_kind == 'concise':
        contract_instance.giveStartingMoney(transact={'from': initiator})
    return


def setter_promise_sell(w3, contract_instance, promiser, value, c_i_broadcast):
    """ function promise(address _promiser, uint256 _value) public """
    value_int = int(value)

    if contract_kind == 'classic':
        tx_hash = contract_instance.transact({'from': promiser, 'to': promiser}).makePromiseOfsell(promiser, value_int)
        receipt = w3.eth.getTransactionReceipt(tx_hash)
        promise_of_sell = contract_instance.call({'from': promiser, 'to': promiser}).getBalance(promiser)

    elif contract_kind == 'concise':
        contract_instance.makePromiseOfsell(promiser, value_int, transact={'from': promiser})
        promise_of_sell = contract_instance.returnPromiseOfsell(promiser)

    return promise_of_sell


def setter_promise_buy(w3, contract_instance, promiser, value, w_j_broadcast):
    """ function promise(address _promiser, uint256 _value) public """
    value_int = int(value)

    if contract_kind == 'classic':
        tx_hash = contract_instance.transact({'from': promiser, 'to': promiser}).makePromiseOfbuy(promiser, value_int)
        receipt = w3.eth.getTransactionReceipt(tx_hash)
        promise_of_buy = 0

    elif contract_kind == 'concise':
        tx_hash = contract_instance.makePromiseOfbuy(promiser, value_int, transact={'from': promiser})
        consumer_receipt = w3.eth.getTransactionReceipt(tx_hash)
        promise_of_buy = contract_instance.returnPromiseOfbuy(w3.eth.accounts[1])


    return promise_of_buy


def setter_burn(w3, contract_instance, consumer, value):
    """ Burn function used by buyers"""
    value_int = int(value)

    if contract_kind == 'classic':
        tx_hash = contract_instance.transact({'from': consumer, 'to': consumer}).consumedEnergy(consumer, value_int)
        receipt = w3.eth.getTransactionReceipt(tx_hash)
        balance_on_bc = 0

    elif contract_kind == 'concise':
        tx_hash = contract_instance.consumedEnergy(consumer, value_int, transact={'from': consumer, 'gas': 300000})
        receipt = w3.eth.getTransactionReceipt(tx_hash)
        balance_on_bc = contract_instance.getBalance(consumer)


    print(balance_on_bc)

    return balance_on_bc


def setter_mint(w3, contract_instance, producer, value):
    """ Mint function available to sellers"""
    value_int = int(value)

    if contract_kind == 'classic':
        tx_hash = contract_instance.transact({'from': producer, 'to': producer}).generatedEnergy(producer, value_int)
        receipt = w3.eth.getTransactionReceipt(tx_hash)
        balance_on_bc = 0

    elif contract_kind == 'concise':
        tx_hash = contract_instance.generatedEnergy(producer, value_int, transact={'from':producer, 'gas': 300000})
        receipt = w3.eth.getTransactionReceipt(tx_hash)
        balance_on_bc = contract_instance.getBalance(producer)

    print(balance_on_bc)

    return balance_on_bc


"""http://web3py.readthedocs.io/en/stable/"""
