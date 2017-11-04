import numpy as np


class BatteryModel:
    # soc = 80  # state of charge

    def __init__(self, Capa, maxChargePower, maxDischargePower, initSoc, dt):
        self.Capa = Capa
        self.maxChargePower = maxChargePower
        self.maxDischargePower = maxDischargePower
        self.initSoc = initSoc
        self.dt = dt
        self.soc = initSoc

    def get_soc(self, battPower):
        soc_change = - np.maximum(np.minimum(battPower, self.maxDischargePower),
                                  self.maxChargePower) * self.dt / self.Capa * 100
        self.soc += soc_change
        return (self.soc)

