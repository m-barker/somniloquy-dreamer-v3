# From https://github.com/allenai/ai2thor-docker/blob/main/example_agent.py
# Tests that cloud rendering works
import ai2thor.controller
import ai2thor.platform
from pprint import pprint


if __name__ == "__main__":
    controller = ai2thor.controller.Controller(
        platform=ai2thor.platform.CloudRendering, scene="FloorPlan28"
    )
    step_count = 0
    while True:
        event = controller.step(action="RotateRight")
        step_count += 1
        print(f"Step: {step_count}")
    pprint(event.metadata["agent"])
