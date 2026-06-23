"""ICSearcher: surrogate-guided fuzzing for UAV autopilot parameters.

Stage-2 package layout (flat modules; stage 4 splits sim/comms into subpackages):
    icsearcher.config         - frozen Config dataclass loaded from data/config.yaml
    icsearcher.logging_config - unified loguru setup
    icsearcher.params         - parameter loading / scaling / Location geometry
    icsearcher.comms          - MAVLink comms + log parsing (DroneMavlink et al.)
    icsearcher.sim            - simulator lifecycle + anomaly detector
    icsearcher.model          - LSTM/TCN surrogate model
    icsearcher.search         - GA fuzzing engine (problem / searcher / fuzzer)
    icsearcher.range          - NSGA-II range derivation (problem / searcher)
"""
__version__ = "0.3.0"
