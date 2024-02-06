from airsim_wrapper import *

print(f"Initializing AirSim...")
aw = AirSimWrapper()
print(f"Done.")

crowd_position = aw.get_position("crowd")
print(crowd_position)

# 去往某个地方之前，必须先 takeoff，因此需要再 prompt 中让 chatgpt 知道这一点
# another useful things:
# before moving the drone to anywhere, we should let the drone take off. Once it token off, we don't need to call takeoff again unless it has already landed.
# Here is an example scenario that if we let the drone take off, it can not move even we let it fly to anywhere.
#
# Me: go to (10, 10, 10)
# You: ```python
# aw.fly_to([10, aw.get_drone_position()[1], aw.get_drone_position()[2]])
# ```
# Me: but, before moving the drone to anywhere, we should let the drone take off.
# you: Apologies for the oversight. You are correct, we should ensure that the drone has taken off before moving it to a specific position. Here's the updated code:
# ```python
# aw.takeoff()
# aw.fly_to([10, aw.get_drone_position()[1], aw.get_drone_position()[2]])
# ```
# Me: go to x=20
# you: ```python
# aw.takeoff()
# aw.fly_to([20, aw.get_drone_position()[1], aw.get_drone_position()[2]])
# ```
# Me: if you already let the drone take off, don't call takeoff again unless it has landed.
# you: Since the drone has already taken off, we can directly use the `aw.fly_to()` function to move the drone to the specified position.
aw.takeoff()
aw.fly_to([10, 10, 10])
