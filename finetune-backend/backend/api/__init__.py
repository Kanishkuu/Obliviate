# Avoid importing a non-existent 'router' from backend.api.routers.
# Expose the routers subpackage for callers that expect backend.api.routers.*
from . import routers

__all__ = ["routers"]