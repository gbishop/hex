from textual.app import App
from textual.containers import Grid, Container, Vertical, Horizontal
from textual.widgets import Button, Static
from sb3_contrib import MaskablePPO
from glob import glob
import numpy as np
from typing import cast
from hexenv import HexEnv

N = 5  # dimension of the Hex Grid
DIR = "MaskablePPO"

fname = sorted(glob(f"{DIR}/*"))[-1]
env = HexEnv(N)
model = MaskablePPO.load(fname, env)


class HexApp(App):
    CSS_PATH = "ui.tcss"

    def compose(self):
        with Vertical():
            yield Static("Your move", id="message")
            with Horizontal():
                yield Button("Replay", id="replay")
                yield Button("Quit", id="quit")

            with Container():
                with Grid():
                    for r in range(N):
                        for _ in range(r):
                            yield Static()
                        for c in range(N):
                            classes = []
                            if (r + c) % 2 == 0:
                                classes.append("even")
                            else:
                                classes.append("odd")
                            yield Button(
                                f"", id=f"i-{r*N+c}", classes=" ".join(classes)
                            )
                        for _ in range(N, 2 * N - r - 1):
                            yield Static()

    def on_mount(self):
        self.message = self.query_one("#message")
        self.reset()

    def reset(self):
        assert isinstance(self.message, Static)
        self.message.update("Your move")
        env.reset()
        for button in self.query("Grid Button"):
            assert isinstance(button, Button)
            button.remove_class("X")
            button.remove_class("O")
            button.label = ""
        self.step()

    def step(self):
        obs = env.game.board
        action_masks = env.game.board == 0
        action, _ = model.predict(
            cast(np.ndarray, obs), action_masks=action_masks, deterministic=True
        )
        obs, reward, _, _, _ = env.step(action)
        button = cast(Button, self.query_one(f"#i-{action}"))
        button.add_class("X")
        button.label = "X"
        if reward == 1:
            assert isinstance(self.message, Static)
            self.message.update("I win!")

    def on_button_pressed(self, event: Button.Pressed):
        id = event.button.id
        if id == "replay":
            self.reset()

        elif id == "quit":
            self.exit()

        else:
            assert isinstance(id, str)
            _, index = id.split("-")
            index = int(index)
            win = env.game.move(index, -1)
            event.button.add_class("O")
            event.button.label = "O"
            if win:
                assert isinstance(self.message, Static)
                self.message.update("You win!")
            else:
                self.step()


if __name__ == "__main__":
    app = HexApp()
    app.run()
