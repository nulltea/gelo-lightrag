//! Integrity-verification placeholder.
//!
//! M3 will add a TwinShield U-Verify-style hash-row check here: the trusted
//! side appends a secret random row `h_A · H` to the activation before
//! offloading, then asserts `h_A · (HW) == hash · W` after recovery. For now
//! this module is intentionally empty so the protocol surface stays open for
//! extension without breaking the M0 substrate API.
