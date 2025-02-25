from manim import *

class SquareToCircle(Scene):
    def construct(self):
        square = Square()  # Create a square
        circle = Circle()  # Create a circle
        circle.set_fill(PINK, opacity=0.5)  # Set the color and transparency of the circle

        self.play(Create(square))  # Animate the creation of the square
        self.play(Transform(square, circle))  # Transform the square into a circle
        self.play(FadeOut(square))  # Fade out the circle
