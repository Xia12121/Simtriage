"""SimTriage: learning when to query the exact game simulator.

The only object that is *evolved* is ``theta`` = a lightweight VoQ predictor (M1)
plus a transferable prototype memory (QVL, M3). The LLM / base policy is frozen.
"""
