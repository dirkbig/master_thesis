import json
import web3

from web3 import Web3, HTTPProvider, TestRPCProvider
from solc import compile_source
from web3.contract import ConciseContract

# Solidity source code
contract_source_code = '''
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
        uint256 public totalSupply = 10000;
        uint256 public startCapital = 1000;
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
        
        
        
        //Initialisation of tokens
         
        function MyToken() public returns(uint256){
            // add require 1-time-give-away check
            supplier = msg.sender;                              // Storing the contract creator's address
            balanceOf[msg.sender] = totalSupply;                // Give the creator all initial tokens
            return totalSupply;
            InitialisedContract(msg.sender, totalSupply);
        }
                
        function giveStartingMoney() external {
            // add require 1-time-give-away check
            balanceOf[msg.sender] = startCapital; 
            InitialisedContract(msg.sender, totalSupply);
        }
        
        
        //Return functions
        function getBalance(address _user) constant returns (uint){
            return balanceOf[_user];
        }
         
            
        function returnPromiseOfsell(address _account) constant returns (uint256){
            return promiseOfsell[_account];
            }
            
        function returnPromiseOfbuy(address _account) constant returns (uint256){
            return promiseOfbuy[_account];
            }

        //Internal transfer
        function _transfer(address _from, address _to, uint _value, uint _class) private {
            require(_to != 0x0);                                        // Prevent transfer to 0x0 address. Use burn() instead
            require(balanceOf[_from] >= _value);                        // Check if the sender has enough
            require(balanceOf[_to] + _value > balanceOf[_to]);          // Check for overflows
            if (_class == 1)
                promise = promiseOfsell[_to] + 1;                       // Check whether agents holds its promise of selling
            if (_class == 2)
                promise = promiseOfbuy[_from] + 1;                      // Check whether agents holds its promise of buying
            require(_value <= promise);                                 // Check whether agents holds its promise of buying
            balanceOf[_from] -= _value;                                 // Subtract from the sender
            balanceOf[_to] += _value;                                   // Add the same to the recipient
            Transfer(_from, _to, _value);                               // Send Event
        }
    
        //External functions
        function transfer(address _to, uint256 _value) public {
            _transfer(msg.sender, _to, _value, class);
        }
    
        //
        function generatedEnergy(address _to, uint256 _value) public {
            _transfer(supplier, _to, _value, classificationOf[msg.sender]);
            CreatedEnergy(_to, _value, balanceOf[_to]);                        // Send Event
        }
    
         
        function consumedEnergy(address _from, uint256 _value) public returns (bool success) {
            // require(_value <= allowance[_from][msg.sender]);     // Check allowance
            // allowance[_from][msg.sender] -= _value;
            _transfer(_from, supplier, _value, classificationOf[msg.sender]);
            // RemovedEnergy(_from, _value);                        // Send Event
            return true;
        }

        
        //make a promise on selling
        function makePromiseOfsell(address _promiser, uint256 _value) public returns (bool success) {
            promiseOfsell[_promiser] = _value;
            promiseOfbuy[_promiser] = 0;
            classificationOf[_promiser] = 1;
            return true;
        }
        
        
        // make a promise on buying
        function makePromiseOfbuy(address _promiser, uint256 _value) public returns (bool success) {
            promiseOfbuy[_promiser] = _value;
            promiseOfsell[_promiser] = 0;
            classificationOf[_promiser] = 2;
            return true;
        }
        
  
    }
'''

compiled_sol = compile_source(contract_source_code)  # Compiled source code
contract_interface = compiled_sol['<stdin>:HouseholdToken']

w3 = Web3(TestRPCProvider())
contract = w3.eth.contract(contract_interface['abi'], bytecode=contract_interface['bin'])
tx_hash = contract.deploy(transaction={'from': w3.eth.accounts[0]})
tx_receipt = w3.eth.getTransactionReceipt(tx_hash)
contract_address = tx_receipt['contractAddress']
contract_instance = w3.eth.contract(contract_interface['abi'], contract_address, ContractFactoryClass=ConciseContract)

# SETTER
""" function MyToken() public returns(uint256){ """
contract_instance.MyToken(transact={'from': w3.eth.accounts[0]})
""" function giveStartingMoney() external {"""
tx_hash = contract_instance.giveStartingMoney(transact={'from': w3.eth.accounts[0]})
consumer_receipt = w3.eth.getTransactionReceipt(tx_hash)
print(consumer_receipt)
""" function makePromiseOfsell(address _promiser, uint256 _value) public"""
contract_instance.makePromiseOfsell(w3.eth.accounts[0], 10000, transact={'from': w3.eth.accounts[0]})

""" function makePromiseOfbuy(address _promiser, uint256 _value) public"""
contract_instance.makePromiseOfbuy(w3.eth.accounts[0], 10000, transact={'from': w3.eth.accounts[0]})

""" function consumedEnergy(address _from, uint256 _value) public returns (bool success) {"""
contract_instance.consumedEnergy(w3.eth.accounts[0], 10, transact={'from': w3.eth.accounts[1], 'gas': 3000000})
""" function generatedEnergy(address _to, uint256 _value) public {"""
contract_instance.generatedEnergy(w3.eth.accounts[0], 10, transact={'from': w3.eth.accounts[0], 'gas': 3000000})



# GETTER
print('Contract value: {}'.format(contract_instance.getBalance(w3.eth.accounts[0])))
print('Contract value: {}'.format(contract_instance.returnPromiseOfsell(w3.eth.accounts[1])))
print('Contract value: {}'.format(contract_instance.returnPromiseOfbuy(w3.eth.accounts[1])))

