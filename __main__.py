import pygame
import numpy as np
from .nn import SimpleNN
import math
from .training_utils import load_weights, save_weights, train_step
from collections import deque
import random

pygame.init()
WIDTH, HEIGHT = 600, 300
win = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 24)
# replay_buffer will likely still be an instance attribute of Game
GRAVITY = 0.5
JUMP_VELOCITY = -10
NUM_AI_PLAYERS = 10 # Reduced for clarity, maybe 10 for original design


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
        self.score_cooldown_timer = 0
        self.last_x_for_reward_logic = self.x # Initialize with current x
        self.consecutive_movement_frames = 0
        self.movement_reward_cooldown = 0
        self.time_since_last_movement_reward = 0
        self.jump_reward_cooldown = 0

    def get_direction_to_score_line(self, score_line_x):
        player_center_x = self.x + self.width / 2
        if player_center_x < score_line_x:
            return -1.0  # Score line is to the right
        elif player_center_x > score_line_x:
            return 1.0   # Score line is to the left
        else:
            return 0.0   # Player is at the score line

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
        self.draw_los(40, -1)
        self.draw_los(40, 1)

    # Modified respawn method to take platforms list
    def respawn(self, platforms): # <--- MODIFIED: Removed start_x, start_y
        chosen_platform = random.choice(platforms) # Pick a random ground platform
        
        # Calculate random x within the chosen platform
        # Ensure the player is fully on the platform
        self.x = random.randint(chosen_platform.rect.left, chosen_platform.rect.right - self.width)
        self.y = chosen_platform.rect.top - self.height # Spawn on top of the platform
        
        self.vx = 0
        self.vy = 0
        self.score = 0 # Reset score here
        self.last_x = self.x # Reset last_x as well
        self.score_cooldown_timer = 0
        self.last_x_for_reward_logic = self.x # Reset to new x after respawn
        self.consecutive_movement_frames = 0
        self.movement_reward_cooldown = 0
        self.time_since_last_movement_reward = 0
        self.jump_reward_cooldown = 0


    def check_score(self, score_line_x):
        center_x = self.x + self.width / 2
        last_center_x = self.last_x + self.width / 2
        
        # The y-coordinate for the top of the ground platforms
        ground_level_y = HEIGHT - 50 # This is still your ground level

        # Determine if the line was crossed
        crossed_line_condition = (last_center_x < score_line_x <= center_x or \
                                  last_center_x > score_line_x >= center_x) and \
                                 (self.y + self.height) < ground_level_y

        if crossed_line_condition and self.score_cooldown_timer == 0:
            self.score += 200
            self.score_cooldown_timer = 30 # Set cooldown for 30 frames

        # Always update last_x to correctly track movement for the next scoring opportunity
        self.last_x = self.x

    def los_input(self, angle_deg, direction, platforms, max_dist=300):
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
        self.is_human = False
        self.FPS = 30
        self.platforms = [
            Platform(0, HEIGHT - 50, 200, 50),
            Platform(260, HEIGHT - 50, 340, 50)
        ]
        self.score_line_x = 230
        self.edge_contact_count = 0
        self.ai_nets = [SimpleNN() for _ in range(NUM_AI_PLAYERS)]
        for idx, net in enumerate(self.ai_nets):
            load_weights(net, suffix=f"_{idx}")

        # Initialize AI players with a random spawn
        self.ai_players = [
            Player(
                x=0, y=0, # These are temporary and will be overridden by respawn
                color=(random.randint(0, 255), # Random Red
                       random.randint(0, 255), # Random Green
                       random.randint(0, 255)) # Random Blue
            )
            for i in range(NUM_AI_PLAYERS)
        ]
        
        for p in self.ai_players:
            p.respawn(self.platforms) # <--- Initial random spawn for AI


        # --- New: Human Player ---
        if self.is_human == True:
            self.human_player = Player(x=0, y=0, color=(255, 255, 0)) # Temporary x,y
            self.human_player.respawn(self.platforms) # <--- Initial random spawn for Human
            self.all_players = self.ai_players + [self.human_player]
            self.human_player_last_score = self.human_player.score 
            self.human_vx_input = 0
            self.human_jump_pressed = False
        else: 
            self.all_players = self.ai_players 

        self.replay_buffer = deque(maxlen=30)
        self.last_scores = [p.score for p in self.ai_players]
       
        self.running = True

    def punish_edge_contact(self, player):

        # Penalize an AI player when touching screen edges.
        for p in self.ai_players:
            if player.x <= 0 or player.x >= WIDTH - player.width:
                self.edge_contact_count += 1
                if self.edge_contact_count == 10:
                    player.score -= 10
                    self.edge_contact_count = 0

    def reset_all_players(self):
        for p in self.ai_players: # <--- Loop and call respawn
            p.respawn(self.platforms)
        if self.is_human == True:
            self.human_player.respawn(self.platforms) # <--- Call respawn for human
            self.human_player_last_score = self.human_player.score

        self.last_scores = [p.score for p in self.ai_players]


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
            if self.is_human == True: # Only process human input if is_human is True
                if event.key == pygame.K_LEFT:
                    self.human_vx_input = -1
                elif event.key == pygame.K_RIGHT:
                    self.human_vx_input = 1
                elif event.key == pygame.K_SPACE:
                    self.human_jump_pressed = True

        elif event.type == pygame.KEYUP:
            # --- Human Player Controls ---
            if self.is_human == True: # Only process human input if is_human is True
                if event.key == pygame.K_LEFT and self.human_vx_input == -1:
                    self.human_vx_input = 0
                elif event.key == pygame.K_RIGHT and self.human_vx_input == 1:
                    self.human_vx_input = 0
                elif event.key == pygame.K_SPACE:
                    self.human_jump_pressed = False


    def update(self):
        # --- Update AI Players ---
        for i, p in enumerate(self.ai_players):
            inputs = np.array([
                p.los_input(40, -1, self.platforms),
                p.los_input(40, 1, self.platforms),
                p.ground_gap_input(self.platforms),
                p.vy / 10.0,  # Normalized vertical velocity
                p.x / 10,
                p.get_direction_to_score_line(self.score_line_x) # Direction to score line
            ])
            '''
            print(p.los_input(40, -1, self.platforms))
            print(p.los_input(40, 1, self.platforms))
            print(p.ground_gap_input(self.platforms))
            print(p.vy / 10)
            '''
            out = self.ai_nets[i].forward(inputs) # + np.random.normal(0, 0.1, size=2)
            self.replay_buffer.append((inputs.copy(), out.copy()))

            if p.on_ground:
                p.vx = out[0] * 3
            if p.on_ground and out[1] > 0.5:
                p.vy = JUMP_VELOCITY
                # JUMP REWARD LOGIC FOR AI
                if p.jump_reward_cooldown == 0:
                    p.score += 5
                    p.jump_reward_cooldown = 150
            
            score_diff = p.score - self.last_scores[i]
            if score_diff != 0:
                gamma = 0.95
                if self.replay_buffer:
                    # Distinguish major events (e.g., clearing a gap) from small
                    # movement rewards so we don't encourage constant jumping
                    MAJOR_REWARD_THRESHOLD = 1
                    if abs(score_diff) >= MAJOR_REWARD_THRESHOLD:
                        for step, (buf_inputs, buf_outputs) in enumerate(reversed(self.replay_buffer)):
                            target = buf_outputs.copy()
                            if score_diff > 0:
                                target[1] = 1.0  # encourage jump on big positive reward
                            discounted = score_diff * (gamma ** step)
                            train_step(self.ai_nets[i], buf_inputs, target, discounted)
                    else:
                        for step, (buf_inputs, buf_outputs) in enumerate(reversed(self.replay_buffer)):
                            discounted = score_diff * (gamma ** step)
                            train_step(self.ai_nets[i], buf_inputs, buf_outputs, discounted)
                    save_weights(self.ai_nets[i], suffix=f"_{i}")
                self.last_scores[i] = p.score

            self.punish_edge_contact(p)

        # --- Update Human Player ---
        if self.is_human == True:
            if self.human_player.on_ground:
                self.human_player.vx = self.human_vx_input * 3
            if self.human_player.on_ground and self.human_jump_pressed:
                self.human_player.vy = JUMP_VELOCITY
                # JUMP REWARD LOGIC FOR HUMAN
                if self.human_player.jump_reward_cooldown == 0:
                    self.human_player.score += 10
                    self.human_player.jump_reward_cooldown = 150
        
        # --- Apply physics and check score for ALL players (AI and Human) ---
        for p in self.all_players:
            # Decrement score cooldown timer
            if p.score_cooldown_timer > 0:
                p.score_cooldown_timer -= 1

            # NEW: Decrement other reward cooldowns
            if p.movement_reward_cooldown > 0:
                p.movement_reward_cooldown -= 1
            if p.jump_reward_cooldown > 0:
                p.jump_reward_cooldown -= 1

            # NEW: Movement reward and penalty logic
            direction_to_line = p.get_direction_to_score_line(self.score_line_x)
            is_moving_towards_line = False
            if direction_to_line == -1 and p.x > p.last_x_for_reward_logic: # Score line to right, player moved right
                is_moving_towards_line = True
            elif direction_to_line == 1 and p.x < p.last_x_for_reward_logic: # Score line to left, player moved left
                is_moving_towards_line = True

            if direction_to_line == 0: # Player is at or beyond the score line relative to their direction of approach
                is_moving_towards_line = False # Not moving 'towards' if already there or past it.

            # Default increment for penalty timer, reset if conditions met.
            p.time_since_last_movement_reward += 1

            if is_moving_towards_line:
                p.consecutive_movement_frames += 1
                p.time_since_last_movement_reward = 0 # Reset if actively moving towards
            else:
                p.consecutive_movement_frames = 0
                # If not moving towards, p.time_since_last_movement_reward continues to increment (already done above)

            if p.movement_reward_cooldown == 0:
                awarded_points = 0
                if p.consecutive_movement_frames >= 20: # Approx 2/3 second of consistent movement
                    awarded_points = 4
                elif p.consecutive_movement_frames >= 10: # Approx 1/3 second
                    awarded_points = 2
                elif p.consecutive_movement_frames >= 5: # Approx 1/6 second
                    awarded_points = 1

                if awarded_points > 0:
                    p.score += awarded_points
                    p.movement_reward_cooldown = 30 # Cooldown for 1 second
                    p.time_since_last_movement_reward = 0 # Reset penalty timer as reward was given
                    p.consecutive_movement_frames = 0 # Reset counter after reward

            # Apply penalty if 30 frames pass without EITHER moving towards the line OR getting a movement category reward
            if p.time_since_last_movement_reward >= 30: # Approx 1 second of no positive action
                p.score -= 4
                p.time_since_last_movement_reward = 0 # Reset penalty timer

            # Update last_x for next frame's comparison for this specific reward logic
            p.last_x_for_reward_logic = p.x

            p.apply_physics(self.platforms)
            
            # Check if player fell off (for both AI and human)
            if p.y > (HEIGHT - 60):
                p.score -= 5000 # Penalize for falling
                p.respawn(self.platforms) # <--- Respawn with random position
            
            p.check_score(self.score_line_x)    

    '''    Jules, leave this comment block alone
            # --- Human Player Scoring update (separate from AI training) ---
        if self.is_human == True:
            human_score_diff = self.human_player.score - self.human_player_last_score
            if human_score_diff != 0:
                self.human_player_last_score = self.human_player.score
    '''

    def draw(self):
        win.fill((0, 0, 0))

        for plat in self.platforms:
            plat.draw()
        pygame.draw.line(win, (255, 255, 0), (self.score_line_x, 0), (self.score_line_x, HEIGHT), 1)

        for p in self.all_players:
            p.draw()

        for i, p in enumerate(self.ai_players):
            txt = font.render(f"AI{i} Score: {p.score}", True, (p.color))
            win.blit(txt, (10, 10 + 20 * i))

        if self.is_human:
            human_txt = font.render(f"Human Score: {self.human_player.score}", True, (255, 255, 0))
            win.blit(human_txt, (10, 10 + 20 * NUM_AI_PLAYERS))

        fps_display_value = int(self.FPS) // 30
        fps_text = font.render(f"FPS: {fps_display_value}X", True, (255, 255, 255))
        # Position in top-right corner with some padding
        # WIDTH - text_width - padding
        fps_text_x = WIDTH - fps_text.get_width() - 10
        fps_text_y = 20
        win.blit(fps_text, (fps_text_x, fps_text_y))

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