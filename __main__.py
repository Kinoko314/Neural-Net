import pygame
import numpy as np
from .nn import SimpleNN
import math
from .training_utils import load_weights, save_weights, train_step
from collections import deque

pygame.init()
WIDTH, HEIGHT = 600, 300
win = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 24)
replay_buffer = deque(maxlen=200)
GRAVITY = 0.5
JUMP_VELOCITY = -10
NUM_AI_PLAYERS = 10  # Number of AI agents

class Platform:
    def __init__(self, x, y, w, h):
        self.rect = pygame.Rect(x, y, w, h)

    def draw(self):
        pygame.draw.rect(win, (100, 100, 100), self.rect)

    def contains_point(self, x, y):
        return self.rect.collidepoint(x, y)

class Player:
    def __init__(self, x, y, color=(0, 255, 0)):
        self.width = 20
        self.height = 30
        self.x = x
        self.y = y
        self.vx = 0
        self.vy = 0
        self.on_ground = False
        self.color = color
        self.score = 0
        self.last_x = x

    def get_rect(self):
        return pygame.Rect(self.x, self.y, self.width, self.height)

    def apply_physics(self, platforms):
        self.vy += GRAVITY
        self.x += self.vx
        self.y += self.vy
        self.on_ground = False
        for plat in platforms:
            player_rect = self.get_rect()
            if player_rect.colliderect(plat.rect):
                if self.vy > 0 and player_rect.bottom - self.vy <= plat.rect.top:
                    self.y = plat.rect.top - self.height
                    self.vy = 0
                    self.on_ground = True
        self.x = max(0, min(WIDTH - self.width, self.x))

    def draw(self):
        pygame.draw.rect(win, self.color, self.get_rect())
        self.draw_los(30, -1)
        self.draw_los(30, 1)

    def respawn(self, start_x, start_y):
        self.x = start_x
        self.y = start_y
        self.vx = 0
        self.vy = 0
        self.score -= 1

    def check_score(self, score_line_x):
        center_x = self.x + self.width / 2
        last_center_x = self.last_x + self.width / 2
        if (last_center_x < score_line_x <= center_x) or (last_center_x > score_line_x >= center_x):
            self.score += 10
        self.last_x = self.x

    def los_input(self, angle_deg, direction, platforms, max_dist=150):
        angle_rad = math.radians(angle_deg)
        dx = math.cos(angle_rad) * direction
        dy = math.sin(angle_rad)
        start_x = self.x if direction == -1 else self.x + self.width
        start_y = self.y
        for d in range(1, max_dist):
            px = int(start_x + dx * d)
            py = int(start_y + dy * d)
            if px < 0 or px >= WIDTH or py >= HEIGHT:
                break
            for plat in platforms:
                if plat.contains_point(px, py):
                    return d / max_dist
        return 1.0

    def draw_los(self, angle_deg, direction):
        angle_rad = math.radians(angle_deg)
        dx = math.cos(angle_rad) * direction
        dy = math.sin(angle_rad)
        start_x = self.x if direction == -1 else self.x + self.width
        start_y = self.y
        for d in range(1, 150, 4):
            px = int(start_x + dx * d)
            py = int(start_y + dy * d)
            if 0 <= px < WIDTH and 0 <= py < HEIGHT:
                if (d // 4) % 2 == 0:
                    win.set_at((px, py), (255, 0, 0))
            else:
                break

    def ground_gap_input(self, platforms, max_dist=2):
        max_px = self.width * max_dist
        step = 2
        for d in range(0, int(max_px), step):
            for offset in [-1, 1]:
                x_check = int(self.x + offset * d)
                y_check = int(self.y + self.height + 1)
                if 0 <= x_check < WIDTH:
                    if not any(p.contains_point(x_check, y_check) for p in platforms):
                        return 1.0 - (d / max_px)
        return 0.0

def run_game():
    FPS = 30
    platforms = [
        Platform(0, HEIGHT - 50, 200, 50),
        Platform(260, HEIGHT - 50, 340, 50)
    ]
    score_line_x = 230

    ai_nets = [SimpleNN() for _ in range(NUM_AI_PLAYERS)]
    for idx, net in enumerate(ai_nets):
        load_weights(net, suffix=f"_{idx}")

    ai_players = [
        Player(x=100, y=HEIGHT - 80, color=(0, 255 - i*5, i*5))
        for i in range(NUM_AI_PLAYERS)
    ]

    def reset_all():
        for i, p in enumerate(ai_players):
            p.respawn(100, HEIGHT - 80 - p.height)

    reset_all()
    running = True
    last_scores = [p.score for p in ai_players]

    while running:
        clock.tick(FPS)
        win.fill((0, 0, 0))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_y:
                    FPS = FPS + 30
                elif event.key == pygame.K_u:
                    if FPS > 30:
                        FPS = FPS - 30
                    else: continue    
                elif event.key == pygame.K_r:
                    reset_all()

        for i, p in enumerate(ai_players):
            inputs = np.array([
                p.los_input(30, -1, platforms),
                p.los_input(30, 1, platforms),
                p.ground_gap_input(platforms),
                1.0
            ])
            out = ai_nets[i].forward(inputs) + np.random.normal(0, 0.1, size=2)
            replay_buffer.append((inputs.copy(), out.copy()))

            p.vx = out[0] * 3
            if p.on_ground and out[1] > 0.8:
                p.vy = JUMP_VELOCITY
            p.apply_physics(platforms)
            p.check_score(score_line_x)

            if p.y > HEIGHT:
                p.respawn(100, HEIGHT - 80 - p.height)

            score_diff = p.score - last_scores[i]
            if score_diff != 0:
                gamma = 0.95
                for j, (inp, act) in enumerate(reversed(replay_buffer)):
                    reward = score_diff * (gamma ** j)
                    train_step(ai_nets[i], inp, act, reward)
                    save_weights(ai_nets[i], suffix=f"_{i}")
                last_scores[i] = p.score

        for plat in platforms:
            plat.draw()
        pygame.draw.line(win, (255, 255, 0), (score_line_x, 0), (score_line_x, HEIGHT), 1)

        for p in ai_players:
            p.draw()

        for i, p in enumerate(ai_players):
            txt = font.render(f"AI{i} Score: {p.score}", True, (255, 255, 255))
            win.blit(txt, (10, 10 + 20 * i))

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    run_game()
