from tina4_python.core import run
from tina4_python.core.server import asgi

# The ASGI application. Point any ASGI server at it:
#   uvicorn   app:app --workers 4
#   hypercorn app:app --workers 4
#   granian   --interface asgi app:app --workers 4
app = asgi()

# `tina4 serve` runs this file directly and takes this path.
if __name__ == "__main__":
    run()
