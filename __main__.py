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
# replay_buffer will likely still be an instance attribute of Game
GRAVITY = 0.5
JUMP_VELOCITY = -10
NUM_AI_PLAYERS = 10 # Reduced for clarity, maybe 10 for original design

# (Platform and Player classes remain the same as before)
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
        self.x = max(0, min(WIDTH - self.width, self.x))  # I think this keeps you from walking off the screen

    def draw(self):
        pygame.draw.rect(win, self.color, self.get_rect())
        self.draw_los(30, -1)
        self.draw_los(30, 1)

    def respawn(self, start_x, start_y):
        self.x = start_x
        self.y = start_y
        self.vx = 0
        self.vy = 0

    def check_score(self, score_line_x):
        center_x = self.x + self.width / 2
        last_center_x = self.last_x + self.width / 2
        # Add condition: Only award points if the player is NOT on the ground
        # or if they are just barely above it (you might need to adjust this threshold)
        if self.y < 180:
            if (last_center_x < score_line_x <= center_x or last_center_x > score_line_x >= center_x):
                self.score += 10
                # print(f"{self.y}")
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


class Game:
    def __init__(self):
        self.FPS = 30
        self.platforms = [
            Platform(0, HEIGHT - 50, 200, 50),
            Platform(260, HEIGHT - 50, 340, 50)
        ]
        self.score_line_x = 230

        self.ai_nets = [SimpleNN() for _ in range(NUM_AI_PLAYERS)]
        for idx, net in enumerate(self.ai_nets):
            load_weights(net, suffix=f"_{idx}")

        self.ai_players = [
            Player(x=100, y=HEIGHT - 80, color=(0, 255 - i*5, i*5))
            for i in range(NUM_AI_PLAYERS)
        ]

        # --- New: Human Player ---
        self.human_player = Player(x=50, y=HEIGHT - 80, color=(255, 255, 0)) # Distinct color
        # Human player's actions will be derived from keyboard input, not NN

        # --- Combine all players into one list for easier iteration for physics/drawing ---
        self.all_players = self.ai_players + [self.human_player]


        self.replay_buffer = deque(maxlen=30)
        
        # self.last_scores needs to track both AI and human scores now
        self.last_scores = [p.score for p in self.ai_players] # Still only for AI for training
        self.human_player_last_score = self.human_player.score

        self.running = True
        self.human_vx_input = 0 # To store horizontal input from keyboard
        self.human_jump_pressed = False # To store jump input from keyboard

    def reset_all_players(self):
        for i, p in enumerate(self.ai_players):
            p.respawn(100, HEIGHT - 80 - p.height)
        self.human_player.respawn(50, HEIGHT - 80 - self.human_player.height) # Respawn human
        self.last_scores = [p.score for p in self.ai_players]
        self.human_player_last_score = self.human_player.score


    def handle_input(self, event):
        if event.type == pygame.QUIT:
            self.running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                self.running = False
            elif event.key == pygame.K_y:
                self.FPS += 30
            elif event.key == pygame.K_u:
                if self.FPS > 30:
                    self.FPS -= 30
            elif event.key == pygame.K_r:
                self.reset_all_players()
            
            # --- Human Player Controls ---
            if event.key == pygame.K_LEFT:
                self.human_vx_input = -1
            elif event.key == pygame.K_RIGHT:
                self.human_vx_input = 1
            elif event.key == pygame.K_SPACE:
                self.human_jump_pressed = True

        elif event.type == pygame.KEYUP:
            # --- Human Player Controls ---
            if event.key == pygame.K_LEFT and self.human_vx_input == -1:
                self.human_vx_input = 0 # Stop moving left
            elif event.key == pygame.K_RIGHT and self.human_vx_input == 1:
                self.human_vx_input = 0 # Stop moving right
            elif event.key == pygame.K_SPACE:
                self.human_jump_pressed = False


    def update(self):
        # --- Update AI Players ---
        for i, p in enumerate(self.ai_players):
            inputs = np.array([
                p.los_input(30, -1, self.platforms),
                p.los_input(30, 1, self.platforms),
                p.ground_gap_input(self.platforms),
            ])
            out = self.ai_nets[i].forward(inputs) + np.random.normal(0, 0.1, size=2)
            self.replay_buffer.append((inputs.copy(), out.copy()))

            p.vx = out[0] * 3
            if p.on_ground and out[1] > 0.8:
                p.vy = JUMP_VELOCITY
            
            # Physics and Scoring for AI players are handled in the general loop below
            # as they are now part of self.all_players

            score_diff = p.score - self.last_scores[i]
            if score_diff != 0:
                gamma = 0.95
                if self.replay_buffer:
                    current_input, current_output = self.replay_buffer[-1]
                    train_step(self.ai_nets[i], current_input, current_output, score_diff)
                    save_weights(self.ai_nets[i], suffix=f"_{i}")
                self.last_scores[i] = p.score

        # --- Update Human Player ---
        # Apply human input to human_player's velocity
        self.human_player.vx = self.human_vx_input * 3 # Control speed like AI
        if self.human_player.on_ground and self.human_jump_pressed:
            self.human_player.vy = JUMP_VELOCITY
        
        # --- Apply physics and check score for ALL players (AI and Human) ---
        for p in self.all_players: # Iterate through the combined list
            p.apply_physics(self.platforms)
            p.check_score(self.score_line_x)

            # Check if player fell off (for both AI and human)
            if p.y > HEIGHT:
                # If it's an AI player, respawn it in its specific spot
                if p in self.ai_players:
                    p.respawn(100, HEIGHT - 80 - p.height)
                    p.score -= 1
                # If it's the human player, respawn in its specific spot
                elif p == self.human_player:
                    p.respawn(50, HEIGHT - 80 - p.height)
                    p.score -= 1


        # --- Human Player Scoring update (separate from AI training) ---
        human_score_diff = self.human_player.score - self.human_player_last_score
        if human_score_diff != 0:
            # You could add human-specific feedback or logging here if desired
            self.human_player_last_score = self.human_player.score


    def draw(self):
        win.fill((0, 0, 0))

        for plat in self.platforms:
            plat.draw()
        pygame.draw.line(win, (255, 255, 0), (self.score_line_x, 0), (self.score_line_x, HEIGHT), 1)

        # Draw all players (AI and Human)
        for p in self.all_players:
            p.draw()

        # Display scores for AI players
        for i, p in enumerate(self.ai_players):
            txt = font.render(f"AI{i} Score: {p.score}", True, (255, 255, 255))
            win.blit(txt, (10, 10 + 20 * i))

        # Display human player score
        human_txt = font.render(f"Human Score: {self.human_player.score}", True, (255, 255, 0))
        win.blit(human_txt, (10, 10 + 20 * NUM_AI_PLAYERS)) # Place it below AI scores

        pygame.display.flip()

    def run(self):
        while self.running:
            clock.tick(self.FPS)

            for event in pygame.event.get():
                self.handle_input(event)

            self.update()
            self.draw()

        pygame.quit()


# --- Main Execution ---
if __name__ == "__main__":
    game = Game()
    game.run()