# --------------------------------------
import inspect

# --------------------------------------
from egomotion import conf

def print_config():
    members = inspect.getmembers(conf)
    for k, v in members:
        if k.startswith("__"):
            continue
        print(f"==[ {str(type(v)):<30} ] {k:>25}: {v}")

if __name__ == "__main__":
    print_config()