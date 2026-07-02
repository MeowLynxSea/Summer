import numpy as np
from pynput import keyboard, mouse
import ctypes

from config import CAMERA_MOUSE_SENSITIVITY, CAMERA_MOVE_SPEED


class FPSCamera:
    def __init__(self):
        self.pos = np.array([0.0, 0.0, -0.6]) 
        self.yaw, self.pitch = 0.0, 0.0
        self.look_at = np.array([0.0, 0.0, 1.0])
        self.right = np.array([1.0, 0.0, 0.0])
        self.keys = set()
        self.mouse_sensitivity = CAMERA_MOUSE_SENSITIVITY
        self.move_speed = CAMERA_MOVE_SPEED
        self.lock_mouse = True
        self.running = True
        self.last_apple_count = -1 
        
        user32 = ctypes.windll.user32
        self.screen_w, self.screen_h = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
        self.center_x, self.center_y = self.screen_w // 2, self.screen_h // 2
        
        self.mouse_ctrl = mouse.Controller()
        self.listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        self.listener.start()

    def _on_press(self, key):
        if key == keyboard.Key.esc: self.running = False
        if key == keyboard.Key.tab: self.lock_mouse = not self.lock_mouse
        try: self.keys.add(key.char.upper())
        except: self.keys.add(key)

    def _on_release(self, key):
        try: self.keys.discard(key.char.upper())
        except: self.keys.discard(key)

    def update(self):
        if self.lock_mouse:
            curr_x, curr_y = self.mouse_ctrl.position
            dx, dy = curr_x - self.center_x, curr_y - self.center_y
            if dx != 0 or dy != 0:
                self.yaw -= dx * self.mouse_sensitivity
                self.pitch -= dy * self.mouse_sensitivity
                self.mouse_ctrl.position = (self.center_x, self.center_y)

        speed = self.move_speed * (5 if keyboard.Key.shift in self.keys else 1)
        if 'W' in self.keys: self.pos += self.look_at * speed
        if 'S' in self.keys: self.pos -= self.look_at * speed
        if 'A' in self.keys: self.pos += self.right * speed
        # D会导致触发OpenNI SDK的保存深度图
        if 'F' in self.keys: self.pos -= self.right * speed
        if 'Q' in self.keys: self.pos[1] -= speed 
        if 'E' in self.keys: self.pos[1] += speed 

        self.pitch = np.clip(self.pitch, -1.5, 1.5)
        front = np.array([np.cos(self.pitch) * np.sin(self.yaw), np.sin(self.pitch), np.cos(self.pitch) * np.cos(self.yaw)])
        self.look_at = front / np.linalg.norm(front)
        self.right = np.array([np.cos(self.yaw), 0, -np.sin(self.yaw)])

    def get_extrinsic(self):
        f_vec, r_vec = self.look_at, self.right
        up_vec = np.cross(r_vec, f_vec)
        R, T = np.eye(4), np.eye(4)
        R[0,:3], R[1,:3], R[2,:3] = r_vec, up_vec, f_vec
        T[:3, 3] = -self.pos
        return R @ T

    def release(self):
        self.listener.stop()