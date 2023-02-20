# # importing the necessary libraries
import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits import mplot3d
# # generating  random dataset
# z = np.random.randint(80, size =(55))
# x = np.random.randint(60, size =(55))
# y = np.random.randint(64, size =(55))
#
# # print(x.size(),y.size(),z.size())
# print(x,y,z)
# # Creating figures for the plot
#
# fig = plt.figure(figsize = (10, 7))
# ax = plt.axes(projection ="3d")
# # Creating a plot using the random datasets
# ax.scatter3D(x, y, z,c = (x+y+z),color = "red")
#
# plt.title("3D scatter plot")
# # display the  plot
# plt.show()
# importing the necessary libraries
from mpl_toolkits import mplot3d
# import matplotlib.pyplot as plt
# import numpy as np
# # Creating random dataset
# z = 4 * np.tan(np.random.randint(10, size =(500))) + np.random.randint(100, size =(500))
# x = 4 * np.cos(z) + np.random.normal(size = 500)
# y = 4 * np.sin(z) + 4 * np.random.normal(size = 500)
# print(x)
# print(y)
# # Creating figure
# fig = plt.figure(figsize = (16, 12))
# ax = plt.axes(projection ="3d")
# # Add x, and y gridlines for the figure
# ax.grid(b = True, color ='blue',linestyle ='-.', linewidth = 0.5,alpha = 0.3)
# # Creating the color map for the plot
# my_cmap = plt.get_cmap('hsv')
# # Creating the 3D plot
# sctt = ax.scatter3D(x, y, z,alpha = 0.8,c = (x + y + z),cmap = my_cmap,marker ='^')
# plt.title("3D scatter plot in Python")
# ax.set_xlabel('X-axis', fontweight ='bold')
# ax.set_ylabel('Y-axis', fontweight ='bold')
# ax.set_zlabel('Z-axis', fontweight ='bold')
# # fig.colorbar(sctt, ax = ax, shrink = 0.6, aspect = 5)
# # display the plot
# plt.show()

def plot_scatter(X, Y, save_dir):  # version 0.5
    from pyecharts import Scatter3D, Page

    Y = np.expand_dims(Y, 1)
    data = np.concatenate((X, Y), axis=1)

    piece = [
        {'value': 0, 'label': 'class A', 'color': '#e57c27'},
        {'value': 1, 'label': 'class B', 'color': '#72a93f'},
        {'value': 2, 'label': 'class C', 'color': '#368dc4'}
    ]

    sc1 = Scatter3D("3D scatter", width=900, height=600)
    sc1.add("", data,
            visual_dimension=3,
            is_visualmap=True,
            is_piecewise=True,
            pieces=piece
            )
    sc1.render(save_dir + '/test.html')

