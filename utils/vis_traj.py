import ast
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import argparse
import ast
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Visualize trajectories from a text file.")
parser.add_argument("txt_file", type=str, help="Path to the text file containing trajectories.")
args = parser.parse_args()

txt_file = args.txt_file

# Load trajectories from file
trajectories = []
with open(txt_file, 'r') as f:
    for line in f:
        s = line.strip()
        if not s:
            continue
        # convert array([...]) to list [...]
        s = s.replace('array([', '[').replace('])', ']')
        pts = ast.literal_eval(s)           # list of [x, y, θ]
        traj = np.array(pts)                # shape (T, 3)

        # Prepend the start pose
        start_pose = np.array([0.4, 0.3, -np.pi/2])
        traj = np.vstack([start_pose, traj])

        trajectories.append(traj)

# Plot
fig, ax = plt.subplots()

size = 0.1
x0, y0 = 0.4, 0.3
goal_x, goal_y = 0.4, 0.0
rect = Rectangle((x0 - size/2, y0 - size/2), size, size,
                 linewidth=1, edgecolor='black', facecolor='none')
goal_rect = Rectangle((goal_x - size/2, goal_y - size/2), size, size,
                      linewidth=1, edgecolor='green', facecolor='none')
ax.add_patch(rect)
ax.add_patch(goal_rect)

# Plot all trajectories in light blue
for traj in trajectories:
    if (traj[-1, 1] >= 0.07 or traj[-1,1] <= -0.03) or (traj[-1, 0] >= 0.46 or traj[-1,0] <= 0.34):
        ax.plot(traj[:, 0], traj[:, 1], color='red', linewidth=1.0)
        ax.plot(traj[:, 0], traj[:, 1], color='lightblue', linewidth=1.0, alpha=0.8)
    else:
        ax.plot(traj[:, 0], traj[:, 1], color='lightblue', linewidth=1.0)

# Compute average trajectory MSE from nominal straight line path
deviations = []
for traj in trajectories:
    nominal_traj_x = np.linspace(0.4, 0.4, num=traj.shape[0])
    nominal_traj_y = np.linspace(0.3, 0.05, num=traj.shape[0])
    nominal_traj = np.vstack([nominal_traj_x, nominal_traj_y]).T
    
    # Compute MSE (Mean Squared Error)
    dists = np.linalg.norm(traj[:, :2] - nominal_traj, axis=1)
    
    # Compute the deviation (Euclidean distance) and store average deviation
    deviations.append(np.mean(dists))

# Calculate and print average MSE and average deviation
average_deviation = np.mean(deviations)
print(f"Average deviation from nominal path {txt_file}: {average_deviation:.4f}\n")

ax.set_aspect('equal', 'box')
ax.set_xlabel('X (m)')
ax.set_xlim(0.1, 0.7)
ax.set_ylim(-0.1, 0.5)
ax.set_ylabel('Y (m)')
plt.show()  # Uncomment to display the plot