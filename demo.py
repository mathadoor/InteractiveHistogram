import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from interactiveHist import Cursor, Distribution

np.random.seed(12345)

df = pd.DataFrame([np.random.normal(32000,200000,3650), 
                   np.random.normal(43000,100000,3650), 
                   np.random.normal(43500,140000,3650), 
                   np.random.normal(48000,70000,3650)], 
                  index=[1992,1993,1994,1995])

#Data Processing
#Generate Random Sample distributions and calculate its mean and standard deviation
samples = 10000
sampling_size = int(0.3*df.shape[1])

means = np.zeros((1,df.shape[0]))
std = np.zeros((1,df.shape[0]))

for i in range(samples):
    nums = np.random.randint(0, df.shape[1]-1, sampling_size)
    mean = np.mean(df.T.iloc[nums,:], axis = 0).values
    means += mean
    std += mean**2

means = means/samples
std = ((std - samples*means**2)/(samples))**0.5


#Creating Bar Chart, setting xticks and the limits
plt.bar([1992, 1993, 1994, 1995], means[0], color = '#c3c3c3', yerr = 1.96*std[0], capsize = 5)
plt.xticks([1992, 1993, 1994, 1995])
plt.xlim((1991.5,1995.5))

#Accessing the artist layer to enforce tighter control 
ax = plt.gca()
fig = plt.gcf()
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylim((0, ( ( max(means[0]) + 2.33*max(std[0]) ) //10000)*10000)  )
ax.set_yticks(list(ax.get_ylim()))

#Creating distribution and cursor object
dist = Distribution(means, std)
cursor = Cursor(ax, dist)

#Connecting backend layer to various events
_ = fig.canvas.mpl_connect('motion_notify_event', cursor.mouse_move)
_ = fig.canvas.mpl_connect('button_press_event', cursor.mouse_click)
_ = fig.canvas.mpl_connect('button_release_event', cursor.mouse_release)
plt.show()