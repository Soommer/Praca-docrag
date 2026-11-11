"""Rejestr profili. Nowy typ dokumentów = nowa klasa Profile dopisana do PROFILES."""
from .base import Profile
from .generic import GenericProfile
from .legal import LegalActProfile

PROFILES: dict[str, type[Profile]] = {
    LegalActProfile.name: LegalActProfile,
    GenericProfile.name: GenericProfile,
}

__all__ = ["PROFILES", "Profile", "GenericProfile", "LegalActProfile"]
