import random


def initialize(agent_value):
    gamma = 0.5
    E_j_supply = random.uniform(0, 100)       # Ej (if sellers)
    E_i_demand = random.uniform(0, 100)       # Ei (if buyers)
    used_energy = 0               # Ei*ci initial
    E_i_allocation = 0
    stored_energy = 0.5
    available_storage = 0
    payment_to_seller = 0
    w_j_storage_factor = random.uniform(0.20, 0.8)


    """initialization function"""
    if agent_value == "E_j_surplus":
        return E_j_supply
    if agent_value == "E_i_demand":
        return E_i_demand
    if agent_value == "payment_to_seller":
        return payment_to_seller
    if agent_value == "w_j_storage_factor":
        return w_j_storage_factor
    if agent_value == "available_storage":
        return available_storage
    if agent_value == "E_i_allocation":
        return E_i_allocation
    if agent_value == "stored_energy":
        return stored_energy
    if agent_value == "used_energy":
        return used_energy
    if agent_value == "gamma":
        return gamma
