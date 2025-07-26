import tkinter as tk
from tkinter import font
import math

from sb3_contrib import MaskablePPO
from glob import glob
import numpy as np
from typing import cast
from hexenv import HexEnv
from argparse import ArgumentParser
from modelmanager import ModelManager


class HexCanvas(tk.Canvas):
    def __init__(self, root, hex_size, rows, cols, border, *args, **kwargs):
        super().__init__(root, *args, **kwargs)
        self.hex_size = hex_size
        self.hex_width = round(math.sqrt(3) * self.hex_size)
        self.hex_height = round(1.5 * self.hex_size)
        self.rows = rows
        self.cols = cols
        self.border = border
        self.cells = []
        self._draw_grid()
        self.bind("<Button-1>", self.on_click)
        self.font = font.Font(family="Arial", size=20, weight="bold")
        X = self.hex_width * (self.cols + 0.5)
        Y = border
        self.message, w, h = self.create_message(X, Y, "Your move")
        Y += h
        self.again, w, h = self.create_button(X, Y, "Again")
        X += w
        self.exit, w, h = self.create_button(X, Y, "Quit")

    def create_button(self, x, y, text):
        padding = 5
        margin = 5
        width = max(self.font.measure(text) + 2 * padding, 100)
        height = self.font.metrics()["linespace"] + 2 * padding
        self.create_rectangle(x, y, x + width, y + height, fill="#eee", width=2)
        id = self.create_text(x + width / 2, y + height / 2, text=text, font=self.font)
        return id, width + margin, height + margin

    def create_message(self, x, y, text):
        padding = 5
        width = self.font.measure(text) + 2 * padding
        height = self.font.metrics()["linespace"] + 2 * padding
        id = self.create_text(x + width / 2, y + height / 2, text=text, font=self.font)
        return id, width, height

    def update_message(self, text):
        self.itemconfig(self.message, text=text)

    def _get_hex_vertices(self, center_x, center_y):
        # Calculate vertices for a pointy-top hexagon
        vertices = []
        for i in range(6):
            angle_deg = 60 * i - 30  # Adjust for pointy-top
            angle_rad = math.pi / 180 * angle_deg
            x = round(center_x + self.hex_size * math.cos(angle_rad))
            y = round(center_y + self.hex_size * math.sin(angle_rad))
            vertices.append((x, y))
        return vertices

    def _draw_grid(self):
        xoffset = self.border + self.hex_width / 2
        yoffset = self.border + self.hex_height / 1.5

        for row in range(self.rows):
            for col in range(self.cols):
                center_x = xoffset + col * self.hex_width
                center_y = yoffset + row * self.hex_height

                center_x += row * self.hex_width / 2

                vertices = self._get_hex_vertices(center_x, center_y)
                self.cells.append(
                    self.create_polygon(vertices, outline="black", fill="#d0d0d0")
                )

        y0 = self.border + self.hex_height * 0.333
        y1 = self.border
        x0 = self.border
        for col in range(self.cols):
            self.create_line(
                [
                    x0 + col * self.hex_width,
                    y0,
                    x0 + (col + 0.5) * self.hex_width,
                    y1,
                    x0 + (col + 0.5) * self.hex_width,
                    y1,
                    x0 + (col + 1.0) * self.hex_width,
                    y0,
                ],
                fill="red",
                width=3,
            )
        y0 = self.border + self.hex_height * self.rows
        y1 = self.border + self.hex_height * self.rows + self.hex_height * 0.333
        x0 = self.border + self.hex_width * (self.rows / 2 - 0.5)
        for col in range(self.cols):
            self.create_line(
                [
                    x0 + col * self.hex_width,
                    y0,
                    x0 + (col + 0.5) * self.hex_width,
                    y1,
                    x0 + (col + 0.5) * self.hex_width,
                    y1,
                    x0 + (col + 1.0) * self.hex_width,
                    y0,
                ],
                fill="red",
                width=3,
            )
        x0 = self.border
        x1 = x0 + self.hex_width / 2
        y0 = self.border + self.hex_height * 0.333
        y1 = y0 + self.hex_height * 0.667
        y2 = y0 + self.hex_height
        for row in range(self.rows):
            self.create_line(
                [
                    x0 + (row * 0.5) * self.hex_width,
                    y0 + row * self.hex_height,
                    x0 + (row * 0.5) * self.hex_width,
                    y1 + row * self.hex_height,
                    x1 + (row * 0.5) * self.hex_width,
                    y2 + row * self.hex_height,
                ],
                fill="blue",
                width=3,
            )
        x0 = self.border + self.hex_width * self.cols
        x1 = x0 + self.hex_width / 2
        y0 = self.border + self.hex_height * 0.333
        y1 = y0 + self.hex_height * 0.667
        y2 = y0 + self.hex_height
        for row in range(self.rows - 1):
            self.create_line(
                [
                    x0 + (row * 0.5) * self.hex_width,
                    y0 + row * self.hex_height,
                    x0 + (row * 0.5) * self.hex_width,
                    y1 + row * self.hex_height,
                    x1 + (row * 0.5) * self.hex_width,
                    y2 + row * self.hex_height,
                ],
                fill="blue",
                width=3,
            )
        row = self.rows - 1
        self.create_line(
            [
                x0 + (row * 0.5) * self.hex_width,
                y0 + row * self.hex_height,
                x0 + (row * 0.5) * self.hex_width,
                y1 + row * self.hex_height,
            ],
            fill="blue",
            width=3,
        )
        self.step()

    def on_click(self, event):
        clicked = self.find_closest(event.x, event.y)[0]
        if clicked in self.cells:
            self.itemconfig(clicked, fill="#0000ff")
            index = self.cells.index(clicked)
            win = env.game.move(index, -1)
            if win:
                self.update_message("You win!")
            else:
                self.step()
        elif clicked == self.exit:
            self.quit()
        elif clicked == self.again:
            self.reset()

    def reset(self):
        for cell in self.cells:
            self.itemconfig(cell, fill="#d0d0d0")
        self.update_message("Your move")
        env.reset()
        self.step()

    def step(self):
        obs = env.game.board
        action_masks = env.game.board == 0
        action, _ = model.predict(
            cast(np.ndarray, obs), action_masks=action_masks, deterministic=True
        )
        obs, reward, _, _, _ = env.step(action)
        self.itemconfig(self.cells[action], fill="#ff0000")
        if reward == 1:
            self.update_message("I win!")


if __name__ == "__main__":
    parser = ArgumentParser(description="Play Hex")
    parser.add_argument(
        "base",
    )
    parser.add_argument("size", type=int)
    args = parser.parse_args()

    manager = ModelManager(args.base, args.size)
    latest, latestpath = manager.latest()

    env = HexEnv(args.size)
    model = manager.load(latest)

    root = tk.Tk()
    root.title("Hex")

    SIZE = args.size
    HEIGHT = 800
    BORDER = 20
    HEX_SIZE = (HEIGHT - 2 * BORDER) / (SIZE * 1.5 + 0.5)
    WIDTH = BORDER * 2 + HEX_SIZE * (math.floor(SIZE * 1.5) * math.sqrt(3))

    canvas = HexCanvas(
        root,
        hex_size=HEX_SIZE,
        rows=SIZE,
        cols=SIZE,
        width=WIDTH,
        height=HEIGHT,
        border=BORDER,
        bg="white",
    )
    canvas.pack()

    root.geometry("+0+0")
    root.mainloop()
