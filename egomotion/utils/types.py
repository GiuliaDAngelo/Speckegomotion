# --------------------------------------
from dataclasses import dataclass

# --------------------------------------
from threading import Event

@dataclass
class Flags:
    attention = Event()
    halt = Event()
