"""Custom, flamo-compatible training criteria for FDN optimization.

torch is imported lazily inside the factory so importing this module (and hence
``pyFDN.train``) stays torch-free.
"""

from __future__ import annotations

from typing import Any


def colorless_sparsity_loss() -> Any:
    """A flamo-compatible sparsity penalty on the FDN feedback matrix.

    Same role and formula as flamo's ``sparsity_loss`` (the colorless penalty of
    *Optimizing Tiny Colorless FDNs*, Dal Santo et al.), but it locates the
    feedback matrix through pyFDN's graph walk
    (:func:`pyFDN.auxiliary.flamo_graph.feedback_matrix_module`), which resolves
    it for **both** a plain ``Series`` core and a ``Parallel`` core (the FDN
    summed with an always-present direct path). flamo's own ``sparsity_loss``
    hard-codes attribute paths that assume a ``.mixing_matrix`` on the Parallel
    branch, so it raises once the direct path is always present.

    The realized matrix ``A = map(param)`` is computed **in-graph** so gradients
    flow back to the matrix parameters. The returned ``nn.Module`` has the
    ``(y_pred, y_target, model)`` call signature flamo's ``Trainer`` uses for
    ``requires_model=True`` criteria (``y_pred``/``y_target`` are ignored).
    """
    import numpy as np
    import torch.nn as nn

    from pyFDN.auxiliary.flamo_graph import feedback_matrix_module

    class _ColorlessSparsity(nn.Module):
        def forward(self, y_pred: Any, y_target: Any, model: Any) -> Any:
            module = feedback_matrix_module(model)
            a = module.map(module.param)
            n = a.shape[-1]
            root_n = float(np.sqrt(n))
            # 0 when |A| is maximally dense (good mixing), 1 when fully sparse.
            return -(a.abs().sum() - n * root_n) / (n * (root_n - 1.0))

    return _ColorlessSparsity()
