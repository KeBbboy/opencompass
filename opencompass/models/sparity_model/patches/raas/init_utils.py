"""Initialization utilities for RaaS."""

# RaaS uses the same forward pass logic as Quest but with access history tracking
# The main difference is in the past_key_value.update_access_history() call
# which is handled directly in the forward pass
