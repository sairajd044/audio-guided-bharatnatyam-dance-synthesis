import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import animation
import numpy as np
import sys
import pickle


def animate(frames, path=None):
    """Add two numbers.

    Args:
        frame (N, J, 3): frames of skeleton keypoints.
        path (str): output path of video

    Returns:
        Plays skeleton video is path is not provided, else video is saved to the path
    """
    fig = plt.figure()
    ax = p3.Axes3D(fig)

    
    joint_names = ['HipCenter', 'Spine', 'ShoulderCenter', 'Head', 'ShoulderLeft', 'ElbowLeft', 
              'WristLeft', 'HandLeft', 'ShoulderRight', 'ElbowRight', 'WristRight', 'HandRight', 
              'HipLeft', 'KneeLeft', 'AnkleLeft', 'FootLeft', 'HipRight', 'KneeRight', 'AnkleRight', 'FootRight']
    parents=[-1,  0,  1,  2,  2,  4,  5,  6,  2,  8,  9, 10,  0, 12, 13, 14,  0, 16, 17, 18]
    # ax.axis("off")

    ax.view_init(elev=90, azim=-90)
    x, y, z = frames.T
    ax.set_xlim(x.min() - 0.1, x.max() + 0.1)
    ax.set_ylim(y.min() - 0.1, y.max() + 0.1)
    ax.set_zlim(z.min() - 0.1, z.max() + 0.1)

    initial_frame = frames[0]
    x, y, z = initial_frame.T

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    # Store joint positions
    points = ax.scatter(x, y, z, color='red', s=10)
    # Store bones as lines
    lines = [] 
    for joint, parent in enumerate(parents):
        if "Left" in joint_names[joint]:
            color = "blue"
        elif "Right" in joint_names[joint]:
            color = "black"
        else:
            color = "orange"
        lines.append(ax.plot([], [], [], color=color)[0])
        
    txt = fig.suptitle('')

    # ax.set_axis_off()
    i = 0

    def update_points(frame):
        nonlocal i
        txt.set_text('frame_num={:d}'.format(i))  # frame number
        i += 1
        x, y, z = frame.T
        # update joint positions
        points._offsets3d = (x, y, z)

        # update bones
        for joint, parent in enumerate(parents):
            if parent != -1:
                lines[joint].set_data(x[[joint, parent]], y[[joint, parent]])
                lines[joint].set_3d_properties(z[[parent, joint]])

        # return modified artists
        return points, txt, lines
    i = 0

    #Make animation
    ani = animation.FuncAnimation(
        fig, update_points, frames=frames[1:], interval=33.33, repeat=False)

    if path != None:
        # Saved video
        writervideo = animation.FFMpegWriter(fps=30)
        ani.save(path, writer=writervideo)
        plt.close()
    else:
        # Show animation
        plt.show()





def main(args):
    path = args[0]
    with open(path, "rb") as f:
        frames = pickle.load(f)
    path = None
    if len(args) == 2:
        path = args[1]
    animate(frames, path)


if __name__ == '__main__':
    main(sys.argv[1:])
