import pygame
import numpy as np

pygame.init()
WIDTH, HEIGHT = 400, 400
win = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

class Block:
    def __init__(self):
        self.x = WIDTH // 2
        self.y = HEIGHT - 40
        self.vel = 5
        self.size = 20

    def draw(self):
        pygame.draw.rect(win, (0, 255, 0), (self.x, self.y, self.size, self.size))

    def move(self, direction):
        if direction == 0:
            self.x -= self.vel
        elif direction == 1:
            self.x += self.vel
        self.x = max(0, min(WIDTH - self.size, self.x))

class SimpleNN:
    def __init__(self):
        self.W1 = np.random.randn(3, 6)
        self.b1 = np.zeros(6)
        self.W2 = np.random.randn(6, 2)
        self.b2 = np.zeros(2)

    def forward(self, x):
        z1 = np.dot(x, self.W1) + self.b1
        a1 = np.tanh(z1)
        z2 = np.dot(a1, self.W2) + self.b2
        return np.argmax(z2)

def run_game():
    block = Block()
    nn = SimpleNN()
    running = True

    while running:
        clock.tick(30)
        win.fill((0, 0, 0))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        nn_input = np.array([block.x / WIDTH, block.y / HEIGHT, 1.0])
        action = nn.forward(nn_input)
        block.move(action)
        block.draw()

        pygame.display.flip()

    pygame.quit()

run_game()
