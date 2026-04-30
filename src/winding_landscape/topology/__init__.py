"""Stage 2: Topology enumeration.

Enumerates valid (Q, 2p, layers, coil_pitch) combinations and computes their
winding-factor harmonic content. Uses SWAT-EM if available, otherwise falls
back to an in-house star-of-slots implementation.
"""
