# myenv/register.py

from gym.envs.registration import register

register(
    id='DroneEnv-v0',
    entry_point='droneenv.drone_env:DroneEnv',
)

register(
    id='VardiyaEnv-v0',
    entry_point='vardiyaenv.vardiya_env:VardiyaEnv',
)
