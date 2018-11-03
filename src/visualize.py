import matplotlib.pyplot as plt

def show_inp(drawings):
    _, axs = plt.subplots(len(drawings),1,figsize=(3,3*len(drawings)))
    for drawing, ax in zip(drawings, axs):
        for x,y in drawing:
            ax.plot(x, y, marker='.')
        ax.axis('off')
        ax.invert_yaxis()