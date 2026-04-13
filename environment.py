import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
import random

class CreatureEnv(gym.Env):
    def __init__(self):
        super().__init__()

        # 20 actions
        self.action_space = spaces.Box(low=-1, high=1, shape=(20,), dtype=np.float32)

        # Increase observation size closer to 172
        self.observation_space = spaces.Box(low=-1, high=1, shape=(120,), dtype=np.float32)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.body_pos = np.array([400.0, 300.0])
        self.body_vel = np.array([0.0, 0.0])

        self.body_angle = 0.0
        self.angular_velocity = 0.0

        # 8 limbs (4 arms + 4 forearms)
        self.limbs = []
        for i in range(8):
            self.limbs.append({
                "angle": random.uniform(-0.3, 0.3),
                "velocity": 0.0,
                "phase": random.uniform(0, 2*math.pi)
            })

        self.goal = np.array([
            random.randint(100, 700),
            random.randint(100, 500)
        ])

        return self._get_obs(), {}

    def _get_obs(self):
        obs = []

        # Body state
        obs.extend(self.body_pos / 800)
        obs.extend(self.body_vel / 50)

        obs.append(math.sin(self.body_angle))
        obs.append(math.cos(self.body_angle))
        obs.append(self.angular_velocity)

        # Goal
        direction = self.goal - self.body_pos
        dist = np.linalg.norm(direction) + 1e-5
        direction_norm = direction / dist

        obs.extend(self.goal / 800)
        obs.extend(direction_norm)
        obs.append(dist / 800)

        # Limb states
        for limb in self.limbs:
            obs.append(limb["angle"])
            obs.append(limb["velocity"])
            obs.append(math.sin(limb["phase"]))
            obs.append(math.cos(limb["phase"]))

        # Fill up to ~120 dims
        while len(obs) < 120:
            obs.append(0.0)

        return np.array(obs, dtype=np.float32)

    def step(self, action):
        # ======================
        # LIMB CONTROL (GAIT)
        # ======================

        total_force = np.array([0.0, 0.0])

        for i in range(8):
            limb = self.limbs[i]

            # Phase progression (important for walking rhythm)
            limb["phase"] += 0.2 + action[i] * 0.1

            # Oscillation motion
            target_angle = math.sin(limb["phase"]) * 0.5

            # Apply torque toward target
            torque = (target_angle - limb["angle"]) * 0.2
            limb["velocity"] += torque
            limb["angle"] += limb["velocity"]

            # Damping
            limb["velocity"] *= 0.85

            # CONTACT FORCE (simulate pushing ground)
            contact_force = max(0, math.cos(limb["phase"]))

            # Convert to direction force
            fx = math.cos(self.body_angle) * contact_force
            fy = math.sin(self.body_angle) * contact_force

            total_force += np.array([fx, fy]) * 0.5

        # ======================
        # BODY MOVEMENT
        # ======================

        # Add RL control influence
        forward = action[16] * 1.5
        rotate = action[17] * 0.05

        move_vec = np.array([
            math.cos(self.body_angle) * forward,
            math.sin(self.body_angle) * forward
        ])

        self.body_vel += move_vec
        self.body_vel += total_force

        # Rotation
        self.angular_velocity += rotate

        # ======================
        # PHYSICS UPDATE
        # ======================

        self.body_pos += self.body_vel
        self.body_vel *= 0.90

        self.body_angle += self.angular_velocity
        self.angular_velocity *= 0.85

        # ======================
        # REWARD (GEOMETRIC)
        # ======================

        direction = self.goal - self.body_pos
        dist = np.linalg.norm(direction) + 1e-5
        direction_norm = direction / dist

        velocity_norm = self.body_vel / (np.linalg.norm(self.body_vel) + 1e-5)

        velocity_match = max(0, np.dot(velocity_norm, direction_norm))

        forward_vec = np.array([
            math.cos(self.body_angle),
            math.sin(self.body_angle)
        ])

        direction_align = max(0, np.dot(forward_vec, direction_norm))

        # Stability bonus (avoid spinning)
        stability = max(0, 1 - abs(self.angular_velocity))

        # Geometric reward
        reward = velocity_match * direction_align * stability

        # ======================
        # TERMINATION
        # ======================

        terminated = dist < 25
        truncated = False

        return self._get_obs(), reward, terminated, truncated, {}