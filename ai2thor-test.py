# From https://github.com/allenai/ai2thor-docker/blob/main/example_agent.py
# Tests that cloud rendering works
import ai2thor.controller
import ai2thor.platform
import time
from pprint import pprint

import functools
import os


def hide_cuda(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Cache the current CUDA_VISIBLE_DEVICES value
        cached_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")

        # Temporarily disable CUDA
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

        try:
            # Call the original function
            return func(*args, **kwargs)
        finally:
            # Restore the original CUDA_VISIBLE_DEVICES value
            os.environ["CUDA_VISIBLE_DEVICES"] = cached_devices

    return wrapper


@hide_cuda
def main():
    controller = ai2thor.controller.Controller(
        scene="FloorPlan28",
    )
    step_count = 0
    start = time.time()
    while True:
        event = controller.step(action="RotateRight")
        step_count += 1
        if step_count % 10 == 0:
            print(f"Time for 10 steps: {time.time() - start}")
            start = time.time()


if __name__ == "__main__":
    main()
