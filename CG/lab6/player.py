from settings import *
import pygame
import math
from map import world_map

class Player:
    def __init__(self):
        self.x, self.y = player_pos
        self.angle = player_angle
        self.collision_radius = TILE // 4  # Минимальное расстояние до стены

    @property
    def pos(self):
        return (self.x, self.y)

    def check_collision(self, x, y):
        """Проверка столкновений с учетом радиуса игрока."""
        for ix in range(-self.collision_radius, self.collision_radius + 1, TILE // 4):
            for iy in range(-self.collision_radius, self.collision_radius + 1, TILE // 4):
                tile = ((x + ix) // TILE * TILE, (y + iy) // TILE * TILE)
                if tile in world_map:  # Если в радиусе есть стена
                    return False
        return True

    def movement(self):
        sin_a = math.sin(self.angle)
        cos_a = math.cos(self.angle)
        keys = pygame.key.get_pressed()
        
        # Проверяем перемещения вперед и назад
        if keys[pygame.K_w]:
            new_x = self.x + player_speed * cos_a
            new_y = self.y + player_speed * sin_a
            if self.check_collision(new_x, self.y):
                self.x = new_x
            if self.check_collision(self.x, new_y):
                self.y = new_y
        if keys[pygame.K_s]:
            new_x = self.x - player_speed * cos_a
            new_y = self.y - player_speed * sin_a
            if self.check_collision(new_x, self.y):
                self.x = new_x
            if self.check_collision(self.x, new_y):
                self.y = new_y

        # Проверяем перемещения влево и вправо
        if keys[pygame.K_a]:
            new_x = self.x + player_speed * sin_a
            new_y = self.y - player_speed * cos_a
            if self.check_collision(new_x, self.y):
                self.x = new_x
            if self.check_collision(self.x, new_y):
                self.y = new_y
        if keys[pygame.K_d]:
            new_x = self.x - player_speed * sin_a
            new_y = self.y + player_speed * cos_a
            if self.check_collision(new_x, self.y):
                self.x = new_x
            if self.check_collision(self.x, new_y):
                self.y = new_y

        # Повороты
        if keys[pygame.K_LEFT]:
            self.angle -= 0.02
        if keys[pygame.K_RIGHT]:
            self.angle += 0.02
