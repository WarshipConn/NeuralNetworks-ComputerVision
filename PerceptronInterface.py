import numpy as np
from tkinter import *

GRID_SIZE = 20
CANVAS_SIZE = 500

txt_path = "C:/Users/henry/OneDrive/Documents/GitHub/NeuralNetworks/PerceptronTrainingData.txt"

grid = np.array([[0]*GRID_SIZE]*GRID_SIZE)

print(grid)

class Square:
    def __init__(self, row, col, canvas):
        self.row = row
        self.col = col
        self.canvas = canvas

        mag = CANVAS_SIZE / GRID_SIZE
        self.gui = canvas.create_rectangle(row*mag, col*mag, (row+1)*mag, (col+1)*mag, fill="black")

        grid[col][row] = 1
    
    def __str__(self):
        return 1

    def delete(self):
        self.canvas.delete(self.gui)
        grid[self.col][self.row] = 0
    

class App:
    def __init__(self, master):
        frame = Frame(master)
        frame.pack(fill=BOTH, expand=1)

        canvas = Canvas(frame, width=CANVAS_SIZE, height=CANVAS_SIZE)
        self.canvas = canvas
        canvas.grid(row=0, column=0, sticky=W+E+N+S)

        self.squares = []
        self.drawing = False
        canvas.bind("<Button-1>", self.mouse_down)
        canvas.bind("<B1-Motion>", self.mouse_move)
        canvas.bind("<ButtonRelease-1>", self.mouse_up)

        menu = Frame(frame, bg="lightblue")
        menu.grid(row=0, column=1, sticky=W+E+N+S)

        self.input = StringVar()
        entry = Entry(menu, textvariable=self.input)
        entry.grid(row=0,column=0)

        store_button = Button(menu, text="Save", command=self.save)
        store_button.grid(row=1, column=0)

        clear_button = Button(menu, text="Clear", command=self.clear)
        clear_button.grid(row=2, column=0)

        clear_all_button = Button(menu, text="ClearAll", command=self.clear_all)
        clear_all_button.grid(row=2, column=1)

        self.result = StringVar()
        status = Label(menu, textvariable=self.result)
        status.grid(row=3, column=0)

    
    def draw_square(self, x, y):
        if x < CANVAS_SIZE and x > 0 and y < CANVAS_SIZE and y > 0:
            row = int(x / CANVAS_SIZE * GRID_SIZE)
            col = int(y / CANVAS_SIZE * GRID_SIZE)

            if grid[col][row] == 0:
                square = Square(row, col, self.canvas)
                self.squares.append(square)

    def mouse_down(self, event):
        self.drawing = True
        self.draw_square(event.x, event.y)
        pass

    def mouse_move(self, event):
        if self.drawing:
            self.draw_square(event.x, event.y)

    def mouse_up(self, event):
        self.drawing = False
        pass


    def save(self):
        print(self.input.get())

        try:
            input = int(self.input.get())

            if input > 9:
                self.input.set("None")
                self.result.set("Input not in domain (0-9)")
                return False
        except ValueError:
            self.input.set("None")
            self.result.set("Invalid input")
            return False

        file = open(txt_path, "a")

        matrix = ""

        for row in grid:
            for v in row:
                matrix += str(v)
            matrix += "\n"
        matrix += "#" + str(self.input.get()) + "\n--\n"

        file.write(matrix)

        file.close()

        self.input.set("None")
        self.result.set("Save successful")
    
    def clear(self):
        for square in self.squares:
            square.delete()
        
        squares = []

        self.input.set("None")
        self.result.set("Canvas cleared")
        print("clear")
    
    def clear_all(self):
        file = open(txt_path, "w")
        file.write("")
        file.close()

        self.result.set("All data cleared.")


root = Tk()
root.minsize(CANVAS_SIZE+300, CANVAS_SIZE)
root.resizable(False, False)
root.wm_title("Perceptron Interface")

app = App(root)

root.mainloop()


