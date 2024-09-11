import numpy as np
import minedojo

env = minedojo.make(
    task_id="harvest_milk_with_empty_bucket_and_cow",
    image_size=(160, 256),
    world_seed=128,
)

print(env.task_prompt)
print(env.task_guidance)
obs = env.reset()
done = False
while not done:
    try:
        action = np.zeros(env.action_space.n)
        next_obs, reward, done, info = env.step(action)
    except:
        continue
