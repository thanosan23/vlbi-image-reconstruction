import tkinter as tk
from tkinter import ttk

from ui import TelescopeApp


def main():
    root = tk.Tk()
    root.title("Radio Interferometer Simulator")

    style = ttk.Style()
    style.theme_use('default')

    window_width = 1400
    window_height = 800
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    center_x = int(screen_width/2 - window_width/2)
    center_y = int(screen_height/2 - window_height/2)
    root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')

    TelescopeApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
