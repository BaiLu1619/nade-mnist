"""Available NADE model variants."""

from nade.models.bernoulli import NADE, bits_per_dimension
from nade.models.categorical import CategoricalNADE
from nade.models.continuous import RNADE

__all__ = ["CategoricalNADE", "NADE", "RNADE", "bits_per_dimension"]
