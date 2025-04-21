# From https://github.com/allenai/ai2thor-docker/blob/main/example_agent.py
# Tests that cloud rendering works
import ai2thor.controller
import ai2thor.platform
import time
from pprint import pprint


if __name__ == "__main__":
    controller = ai2thor.controller.Controller(
        platform=ai2thor.platform.CloudRendering,
        scene="FloorPlan28",
    )
    step_count = 0
    start = time.time()
    while True:
        event = controller.step(action="RotateRight")
        step_count += 1
        if step_count % 100 == 0:
            print(f"Time for 100 steps: {time.time() - start}")
            start = time.time()
    pprint(event.metadata["agent"])
