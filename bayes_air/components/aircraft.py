"""Define a class representing an aircraft."""
from dataclasses import dataclass

from bayes_air.components.types import TailNumber
from bayes_air.schedule import ItineraryItem


@dataclass
class Aircraft:
    tail_number: TailNumber
    itinerary: list[ItineraryItem]
    _current_itinerary_item_idx: int = 0

    @property
    def current_leg(self) -> ItineraryItem:
        return self.itinerary[self._current_itinerary_item_idx]

    @property
    def next_leg(self) -> ItineraryItem:
        if self._current_itinerary_item_idx == len(self.itinerary) - 1:
            return None
        else:
            return self.itinerary[self._current_itinerary_item_idx + 1]

    def advance_itinerary(self) -> None:
        # Only advance if we're not on the last leg
        if self._current_itinerary_item_idx < len(self.itinerary) - 1:
            self._current_itinerary_item_idx += 1

    def __str__(self) -> str:
        return (
            f"{self.tail_number}_"
            f"{self.current_leg.origin}->{self.current_leg.destination}"
        )
