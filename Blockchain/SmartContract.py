import json
import web3

from web3 import Web3, HTTPProvider, TestRPCProvider
from solc import compile_source
from web3.contract import ConciseContract

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
    
    interface energyToken { function receiveApproval(address _from, uint256 _value, address _token, bytes _extraData); }
    
    contract HouseholdToken {
        // Public variables of the token
        string public name = "Energy Bazaar";
        string public symbol = "EBZ";
        uint8 public decimals = 100;
        uint256 public totalSupply = 1000000;
        // Private account of the contract creator. to be used as a base account
        address supplier;
    
        // This creates an array with all balances
        // The actual balance
        mapping (address => uint256) public balanceOf;
        // The balance allowed for the Smart Meter to spend
        mapping (address => mapping (address => uint256)) public allowance;
    
        // This generates a public event on the blockchain that will notify clients
        event Transfer(address indexed from, address indexed to, uint256 value);
    
        // This notifies clients about the amount donated
        event Donate(address indexed from, uint256 value);
    
        //This notifies energy tokens created
        event CreatedEnergy(address indexed from, uint256 value);
    
        //This notifies energy tokens used
        event RemovedEnergy(address indexed from, uint256 value);
    
        //This notifies community on initialisation of contract through MyToken() function
        // event InitialisedContract();
        
        /**
         * Constructor function
         *
         * Initialises contract with initial supply tokens to the creator of the contract
         */
         
        function MyToken() {
            supplier = msg.sender;                              // Storing the contract creator's address
            balanceOf[msg.sender] = totalSupply;                // Give the creator all initial tokens
        }
    
        /**
         * Internal transfer, only can be called by this contract
         */
         
        function _transfer(address _from, address _to, uint _value) internal {
            require(_to != 0x0);                                // Prevent transfer to 0x0 address. Use burn() instead
            require(balanceOf[_from] >= _value);                // Check if the sender has enough
            require(balanceOf[_to] + _value > balanceOf[_to]);  // Check for overflows
            balanceOf[_from] -= _value;                         // Subtract from the sender
            balanceOf[_to] += _value;                           // Add the same to the recipient
            Transfer(_from, _to, _value);                       //Send Event
        }
    
        /**
         * Transfer tokens from one account to other(s)
         *
         * Send `_value` tokens to `_to` from your account
         *
         * @param _to The address of the recipient
         * @param _value the amount to send
         */
        function transfer(address _to, uint256 _value) {
            _transfer(msg.sender, _to, _value);
        }
    
        /**
         * Add tokens to the energy creators account
         *
         * Send `_value` tokens to `_to` from `supplier` account
         *
         * @param _to The address of the person owning Smart Meter
         * @param _value the amount to send
         */
        function createdEnergy(address _to, uint256 _value) {
            _transfer(supplier, _to, _value);
            CreatedEnergy(_to, _value);                                 // Send Event
        }
    
    
        /**
         * Transfer tokens from other address (i.e. Smart meter)
         * To be triggered by smart meter
         *
         * Send `_value` tokens to `owner` in behalf of `_from`
         *
         * @param _from The address of the sender
         * @param _value the amount to send
         */
        function transferFrom(address _from, uint256 _value) returns (bool success) {
            require(_value <= allowance[_from][msg.sender]);     // Check allowance
            allowance[_from][msg.sender] -= _value;
            _transfer(_from, supplier, _value);
            RemovedEnergy(_from, _value);                        // Send Event
            return true;
        }
    
        /**
         * Set allowance for Smart Meter to spend on your behalf
         *
         * Allows `_spender` to spend no more than `_value` tokens in your behalf
         *
         * @param _spender The smart meter address authorized to spend
         * @param _value the max amount they can spend
         */
        function approve(address _spender, uint256 _value)
            returns (bool success) {
            allowance[msg.sender][_spender] = _value;
            return true;
        }
    
        /**
         * Donate Tokens to the platform for social service/impact
         *
         * Remove `_value` tokens from the system irreversibly
         *
         * @param _value the amount of money to burn
         */
        function donate(uint256 _value) returns (bool success) {
            require(balanceOf[msg.sender] >= _value);   // Check if the sender has enough
            _transfer(msg.sender, supplier, _value);
            Donate(msg.sender, _value);                   // Send Event
            return true;
        }    
    }
    """

    """ compile Solidity code """
    compiled_sol = compile_source(contract_source_code) # Compiled source code
    contract_interface = compiled_sol['<stdin>:HouseholdToken']

    """ create web3.py test environment instance"""
    w3 = Web3(TestRPCProvider())

    # for i in range(len(w3.eth.accounts)):
    #     print(w3.eth.accounts[i])

    return contract_interface, w3


def deploy_SC(contract_interface, w3):
    """ Deploy smart_contract: : The Bazaar is opened"""

    """ use the web3.eth.contract(...) method to generate the contract factory classes for your contracts, 
        instantiate contract"""
    contract = w3.eth.contract(contract_interface['abi'], bytecode=contract_interface['bin'])

    deployment_tx_hash = contract.deploy(transaction={'from': w3.eth.accounts[0], 'gas': 4000000})

    tx_receipt = w3.eth.getTransactionReceipt(deployment_tx_hash)
    contract_address = tx_receipt['contractAddress']

    contract_instance = w3.eth.contract(contract_interface['abi'], contract_address,
                                        ContractFactoryClass=ConciseContract)

    return w3, contract_instance, deployment_tx_hash, contract_address



    # # Get transaction hash from deployed contract
    # print('deployment_tx_hash')
    # # Get tx receipt to get contract address
    # print('Contract deployed, contract address:', contract_address)
    # print(contract_address)
    # print(w3.eth.accounts)


def setters(w3, contract_instance, deployment_tx_hash, contract_address):
    """
    constructor functions
        function MyToken()

    sc functions
        function _transfer(address _from, address _to, uint _value)
        function transfer(address _to, uint256 _value)
        function mintToken(address target)
        function burnFrom(address _from)

        function approve(address _spender, uint256 _value)
        function donate(uint256 _value)
    """

    contract_instance.MyToken(transact={'from': w3.eth.accounts[1]})
    # transfer_filter = contract_instance.eventFilter('InitialisedContract')
    # transfer_filter.get_new_entries()


    # contract_instance.mintToken(w3.eth.accounts[1])
    # contract_instance.mintToken(w3.eth.accounts[1])
    #
    # contract_instance.transfer(w3.eth.accounts[2], 1, transact={'from': w3.eth.accounts[1]})


    # Contract instance in concise mode
    #
    # """ Getters + Setters for web3.eth.contract object """
    # print('Contract value: {}'.format(contract_instance.greet()))
    # contract_instance.setGreeting('Nihao', transact={'from': w3.eth.accounts[0]})
    # print('Setting value to: Nihao')
    # print('Contract value: {}'.format(contract_instance.greet()))
    #









"""http://web3py.readthedocs.io/en/stable/"""
