"""
TODO:

    Kwh as y-axis does NOT make sense since the plots look continuous while it is discreet. So either discretize the plot or
    convert Kwh per 10 minutes to kW

    Plot sense of fairness as standard deviation from average deficit. To what degree deficit is spread-out over agents or
        whether there are extreme cases where agents are totally left out.

    Show E_demand (mean/std) with or w/o EnergyBazaar: show that E_demand is bended
        expected: w/o EBZ; E_demand is only high when no production. with EBZ, E_demand should
        be higher during production (of consumers? or also of prosumers?)
        Maybe show as well with continuous load? Makes it more clear

    w_mean stabilizes at 0.8: show that when distortion or sudden higher/lower load, w_mean restabilizes to an other value
    probably run parameter sweep over lambda?

    Express EBZ performance in the amount of kwh imported from the grid:
    both with and w/o EBZ, depletion events arise, so in either case, MG has to import extra energy. With EBZ, this amount
    should be way lower. SHOW this and the welfare of a typical prosumers in both cases.
    Since zero-sum game, there are always winners and losers within the community. but the total welfare of the community can also be expressed

    Topology of Sergio Grammatico:  Network Aggregative Games and Distributed Mean Field Control via Consensus Theory


    Utility function Sellers without prediction; yes, but soc_preferred is still model-predictive! change this aswell otherwise EBZ gets pwned!

    Bug:  agent.soc_influx = agent.E_i_demand + agent.deficit in microgrid model when battery is depleting? why agent.E_i_demand??!


DONE:

    Extend number of agent to 100
    Shuffling agents at each optimization game (only microgrid_model for now)
    Max discharge rate constraint of batteries; implement this // ESSENTIAL!
    update w_sharing_factor of agents to their actual supplied energy: oversupply is a weird thing
    X-axis instead of steps make it minutes (or time for that matter)
    show plots for consumers or prosumers:: averages cannot be taken between the two: seperate them
    Depletion Event comparison with and w/o EnergyBazaar sharing: deficit measuments
    Parameter survey/sweep on battery size wrt depletion events: decided upon: 15kwh
    Time-stamp state in the smart contract for async optimzation/ random possibility where agents do not update the blockchain
    creating asycn. then time-stamp can be used to either ignore that agent or optimize with it anyways
    Play with battery prediction horizon: currently only around 70 steps: a hour.
        increase to 700 steps: 10 hours
        show prediction has a positive effect; if not in obvious data, what functionality is enabled by it?





"""
