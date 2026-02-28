"""Auxiliary modules (utils, acoustics, allpass, flamo wrappers, etc.)."""
from .utils import skew
from .flamo import gain_module, delay_module, sos_filter_module
from .allpass import (
    poletti_allpass,
    series_allpass,
    nested_allpass,
    is_uniallpass,
    is_allpass,
    is_paraunitary,
)
from .flamo_graph import flamo_model_to_nodes, flamo_nodes_flat, draw_flamo_graph
from .flamo_probe import FlamoGraphZFilter, flamo_graph_to_zfilter, probe_flamo_z
from .flamo_autograd_probe import (
    FlamoAutogradGraphZFilter,
    attach_autograd_probe,
    flamo_graph_to_autograd_zfilter,
    probe_flamo_z_autograd,
)
from .flamo_runtime_probe import (
    has_flamo_native_probe,
    probe_flamo_runtime,
    probe_flamo_recursion_runtime,
)

__all__ = [
    "skew",
    "gain_module",
    "delay_module",
    "sos_filter_module",
    "poletti_allpass",
    "series_allpass",
    "nested_allpass",
    "is_uniallpass",
    "is_allpass",
    "is_paraunitary",
    "flamo_model_to_nodes",
    "flamo_nodes_flat",
    "draw_flamo_graph",
    "probe_flamo_z",
    "FlamoGraphZFilter",
    "flamo_graph_to_zfilter",
    "probe_flamo_z_autograd",
    "FlamoAutogradGraphZFilter",
    "flamo_graph_to_autograd_zfilter",
    "attach_autograd_probe",
    "has_flamo_native_probe",
    "probe_flamo_runtime",
    "probe_flamo_recursion_runtime",
]
