from ..ssh import SSHHost
from .base import Provider, Target, load_target, parse_target
from .ssh import SSHProvider
from .vast import VastProvider


PROVIDERS = {SSHProvider.name: SSHProvider, VastProvider.name: VastProvider}
DEFAULT_PROVIDER = SSHProvider.name


def create_provider(target):
    """Build the provider for a target, including its SSH transport."""
    try:
        provider_class = PROVIDERS[target.provider]
    except KeyError:
        raise ValueError(f"Unknown provider '{target.provider}'; supported: "
                         + ", ".join(sorted(PROVIDERS))) from None
    transport = SSHHost(target.host, target.key, target.port, target.known_hosts)
    return provider_class(transport)


__all__ = ["DEFAULT_PROVIDER", "PROVIDERS", "Provider", "Target", "create_provider",
           "load_target", "parse_target"]
