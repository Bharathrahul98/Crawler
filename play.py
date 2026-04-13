import pygame
import math
from stable_baselines3 import PPO
from environment import CreatureEnv

pygame.init()
screen = pygame.display.set_mode((800, 600))
clock = pygame.time.Clock()

env = CreatureEnv()
model = PPO.load("crawler_model")

obs, _ = env.reset()

def draw(env):
    x, y = int(env.body_pos[0]), int(env.body_pos[1])

    # ======================
    # BODY (MAIN)
    # ======================
    pygame.draw.circle(screen, (50, 150, 255), (x, y), 20)

    # HEAD (top)
    pygame.draw.circle(screen, (255, 200, 0), (x, y - 10), 8)

    # ======================
    # DIRECTION RING
    # ======================
    pygame.draw.circle(screen, (0, 100, 255), (x, y), 30, 2)

    # Forward arrow
    fx = x + math.cos(env.body_angle) * 35
    fy = y + math.sin(env.body_angle) * 35
    pygame.draw.line(screen, (0, 200, 0), (x, y), (int(fx), int(fy)), 3)

    # ======================
    # LIMBS (4 arms + 4 forearms)
    # ======================
    for i, limb in enumerate(env.limbs):
        base_angle = limb["angle"] + i * (math.pi / 4)

        # ARM (upper)
        lx = x + math.cos(base_angle) * 30
        ly = y + math.sin(base_angle) * 30

        pygame.draw.line(screen, (200, 200, 200), (x, y), (int(lx), int(ly)), 5)

        # JOINT
        pygame.draw.circle(screen, (255, 255, 255), (int(lx), int(ly)), 4)

        # FOREARM (lower)
        fx = lx + math.cos(base_angle) * 25
        fy = ly + math.sin(base_angle) * 25

        pygame.draw.line(screen, (100, 150, 255), (int(lx), int(ly),), (int(fx), int(fy)), 5)

    # ======================
    # GOAL (CUBE STYLE)
    # ======================
    gx, gy = int(env.goal[0]), int(env.goal[1])

    size = 30

    pygame.draw.rect(screen, (0, 200, 0), (gx - size//2, gy - size//2, size, size))

    # cube shading (simple 3D feel)
    pygame.draw.line(screen, (0, 255, 0), (gx - 15, gy - 15), (gx - 10, gy - 20), 2)
    pygame.draw.line(screen, (0, 255, 0), (gx + 15, gy - 15), (gx + 20, gy - 20), 2)

running = True
while running:
    screen.fill((255, 255, 255))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    action, _ = model.predict(obs)
    obs, _, done, _, _ = env.step(action)

    draw(env)

    if done:
        obs, _ = env.reset()

    pygame.display.flip()
    clock.tick(60)

pygame.quit()