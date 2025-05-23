import pygame
from settings import *
from ray_casting import ray_casting
from map import mini_map

class Drawing:
    def __init__(self, sc, sc_map):
        self.sc = sc
        self.sc_map = sc_map
        self.font = pygame.font.SysFont('Arial', 36, bold=True)
        self.textures = {'1': pygame.image.load('img/1.jpg').convert(),
                         '2': pygame.image.load('img/2.jpg').convert(),
                         'S': pygame.image.load('img/sky.png').convert()
                         }

    def background(self, angle):
        sky_offset = -5 * math.degrees(angle) % WIDTH
        self.sc.blit(self.textures['S'], (sky_offset, 0))
        self.sc.blit(self.textures['S'], (sky_offset - WIDTH, 0))
        self.sc.blit(self.textures['S'], (sky_offset + WIDTH, 0))
        pygame.draw.rect(self.sc, DARKGRAY, (0, HALF_HEIGHT, WIDTH, HALF_HEIGHT))

    def world(self, player_pos, player_angle):
        ray_casting(self.sc, player_pos, player_angle, self.textures)

    def fps(self, clock):
        display_fps = f'FPS: {int(clock.get_fps())}'
        render = self.font.render(display_fps, True, WHITE)

        # Создание рамки для счётчика FPS
        fps_bg_rect = render.get_rect(topleft=(10, 10))
        fps_bg_rect.inflate_ip(10, 5)  # Увеличение размера рамки
        pygame.draw.rect(self.sc, BLACK, fps_bg_rect)
        pygame.draw.rect(self.sc, WHITE, fps_bg_rect, 2)  # Граница рамки

        # Отображение FPS
        self.sc.blit(render, (fps_bg_rect.x + 5, fps_bg_rect.y + 2))


    def mini_map(self, player):
        self.sc_map.fill(BLACK)
        map_x, map_y = player.x // MAP_SCALE, player.y // MAP_SCALE
        pygame.draw.line(self.sc_map, YELLOW, (map_x, map_y), (map_x + 12 * math.cos(player.angle),
                                                 map_y + 12 * math.sin(player.angle)), 2)
        pygame.draw.circle(self.sc_map, RED, (int(map_x), int(map_y)), 5)
        for x, y in mini_map:
            pygame.draw.rect(self.sc_map, SANDY, (x, y, MAP_TILE, MAP_TILE))
        self.sc.blit(self.sc_map, MAP_POS)