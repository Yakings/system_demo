# How to Create a GUI in Python

# Import modules
import tkinter as tk  # module to create a GUI
from PIL import ImageTk  # module to add a background picture
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

root = tk.Tk()

canvas1 = tk.Canvas(root, width=1000, height=350)  # create the canvas (tkinter module)
canvas1.pack()

imagepath1 = r'G:\我的照片\837.jpg'  # include the path for the image (use 'r' before the path string to address any special character such as '\'. Also, make sure to include the image type - here it is jpg)
image1 = ImageTk.PhotoImage(file=imagepath1)  # PIL module
canvas1.create_image(700, 250, image=image1)  # create a canvas image to place the background image

button1 = tk.Button(root, text='Exit Application',
                    command=root.destroy)  # create an Exit button to close the main window
canvas1.create_window(85, 300,
                      window=button1)  # create a canvas window to place the Exit Application button on top of the background image

entry1 = tk.Entry(root)  # create 1st entry box
canvas1.create_window(100, 200,
                      window=entry1)  # create a canvas window to place the 1st entry box on top of the background image

entry2 = tk.Entry(root)  # create 2st entry box
canvas1.create_window(100, 220,
                      window=entry2)  # create a canvas window to place the 2nd entry box on top of the background image

entry3 = tk.Entry(root)  # create 3st entry box
canvas1.create_window(100, 240,
                      window=entry3)  # create a canvas window to place the 3rd entry box on top of the background image


def insert_number():  # create a function/command to be called by the button (i.e., button2 below)
    global x1  # add 'global' before the variable x1, so that you can use that variable outside of the command/function if ever needed
    global x2  # add 'global' before the variable x2, so that you can use that variable outside of the command/function if ever needed
    global x3  # add 'global' before the variable x3, so that you can use that variable outside of the command/function if ever needed
    x1 = float(
        entry1.get())  # store the data input by the user as a variable x1 (float is added to handle any formatting issues)
    x2 = float(
        entry2.get())  # store the data input by the user as a variable x2 (float is added to handle any formatting issues)
    x3 = float(
        entry3.get())  # store the data input by the user as a variable x3 (float is added to handle any formatting issues)

    # create a bar chart once the variables x1, x2 and x3 are inserted by the user (and the user then clicks on button2 below)
    figure1 = Figure(figsize=(5, 4), dpi=100)  # create a Figure (matplotlib module)
    subplot1 = figure1.add_subplot(111)  # add a subplot
    xAxis = [float(1), float(2),
             float(3)]  # intakes the values inserted under x1, x2 and x3 to represent the x Axis
    yAxis = [float(x1), float(x2),
             float(x3)]  # intakes the values inserted under x1, x2 and x3 to represent the y Axis
    subplot1.bar(xAxis, yAxis, color='g')  # create the bar chart based on the input variables x1, x2, and x3
    bar1 = FigureCanvasTkAgg(figure1, root)  # create a canvas figure (matplotlib module)
    bar1.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=0)

    # create a pie chart once the variables x1, x2 and x3 are inserted by the user (and the user then clicks on button2 below)
    figure2 = Figure(figsize=(5, 4), dpi=100)  # create a Figure (matplotlib module)
    subplot2 = figure2.add_subplot(111)  # add a subplot
    labels2 = 'Label1', 'Label2', 'Label3'  # add labels for each slice in the pie chart
    pieSizes = [float(x1), float(x2),
                float(x3)]  # intakes the values inserted under x1, x2 and x3 to represent the pie slices
    explode2 = (0, 0.1, 0)  # explodes the 2nd slice (i.e. 'Label2')
    subplot2.pie(pieSizes, explode=explode2, labels=labels2, autopct='%1.1f%%', shadow=True,
                 startangle=90)  # create the pie chart based on the input variables x1, x2, and x3
    subplot2.axis('equal')  # Use equal to draw the pie chart as a circle
    pie2 = FigureCanvasTkAgg(figure2, root)  # create a canvas figure (matplotlib module)
    pie2.get_tk_widget().pack()


button2 = tk.Button(root, text='Click to Draw Chart ',
                    command=insert_number)  # button to call the 'insert_number' command above
canvas1.create_window(97, 270, window=button2)  # create a canvas window to place button2 on top of the background image

root.mainloop()