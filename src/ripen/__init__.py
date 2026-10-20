"""ripen: distributional shelf-life prediction from produce photos.

Every item (one avocado, one tomato, ...) is a sequence of visits. Each visit
has a time, an optional ripeness stage label, and optional image features.
The target is the spoilage time tau, which is only ever known to lie in an
interval (last visit seen unspoiled, first visit seen spoiled], or to exceed
the last visit (right-censored). Every model in :mod:`ripen.models` maps a
visible prefix of visits to a distribution over tau and is scored by the same
code in :mod:`ripen.eval`.
"""

__version__ = "0.2.0"
