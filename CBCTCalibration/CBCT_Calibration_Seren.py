import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches, interactive
from matplotlib.patches import Ellipse
from scipy.optimize import curve_fit, minimize
from scipy.stats import zscore, scoreatpercentile
from tqdm import tqdm


# === Normalization Function ===
def normalize_data(img, ob, dc):
    print("[INFO] Starting normalization...")
    ob = ob - dc
    ob[ob < 1] = 1
    lcal = np.empty_like(img)
    for idx in range(img.shape[0]):
        tmp = img[idx] - dc
        tmp[tmp <= 0] = 1
        lcal[idx] = tmp / ob
    lcal = -np.log(lcal)
    print("[INFO] Normalization completed. Value range: [{:.3f}, {:.3f}]".format(np.min(lcal), np.max(lcal)))
    return lcal


# === Baseline Removal Function ===
def remove_baseline(img):
    print("[INFO] Removing baseline...")
    baseline = img.mean(axis=0).mean(axis=0).reshape(1, -1)
    b2 = np.matmul(np.ones([img.shape[1], 1]), baseline)
    res = img.copy()
    for idx in range(img.shape[0]):
        res[idx] -= b2
    print("[INFO] Baseline removed. Value range: [{:.3f}, {:.3f}]".format(np.min(res), np.max(res)))
    return res


# === Load Data ===
print("[INFO] Loading input files...")
dc = np.load("dc.npy")
ob = np.load("ob.npy")
proj = np.load("proj.npy")
print(f"[INFO] Shapes - DC: {dc.shape}, OB: {ob.shape}, Proj: {proj.shape}")

# === Show Dark Field, Bright Field, and a Sample Projection ===
# fig, ax = plt.subplots(1, 3, figsize=(12, 4))
# ax[0].imshow(dc, cmap='magma')
# ax[0].set_title('Dark Current (DC)')
# a1 = ax[1].imshow(ob, cmap='magma')
# fig.colorbar(a1, ax=ax[1])
# ax[1].set_title('Open Beam (OB)')
# a2 = ax[2].imshow(proj[1], cmap='magma')
# fig.colorbar(a2, ax=ax[2])
# ax[2].set_title('Sample Projection')
# plt.tight_layout()
# plt.show()

# === Normalize and Optional Flip ===
start_time = time.time()
lcal = normalize_data(proj, ob, dc)
print("[INFO] Normalization time: {:.2f} s".format(time.time() - start_time))

flip_projection = True
if flip_projection:
    print("[INFO] Vertical flip enabled.")
    lcal = lcal[:, ::-1, :]

# === Show Histogram and Projection Slice ===
# fig, ax = plt.subplots(1, 2, figsize=(10, 4))
# ax[0].imshow(lcal[1], vmin=0, vmax=3, cmap='magma')
# fig.colorbar(ax[0].images[0], ax=ax[0])
# ax[0].set_title("Normalized Projection")
# ax[1].hist(lcal[1].ravel(), bins=1000, color='skyblue')
# ax[1].set_xlim(0, 2.5)
# ax[1].set_title("Histogram")
# plt.tight_layout()
# plt.show()

print("[INFO] Normalized shape:", lcal.shape)

# === Before Baseline Removal ===
idx = 2
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(lcal[idx], vmin=0, vmax=4, cmap='magma')
plt.title("Before Baseline Removal")
plt.colorbar()
plt.subplot(1, 2, 2)
plt.plot(lcal[idx].mean(axis=0)[:1750], label='Row Mean')
plt.plot(lcal.mean(axis=0).mean(axis=0)[:1750], label='Baseline')
plt.title("Horizontal Profile")
plt.legend()
plt.tight_layout()
plt.show()

# === Remove Baseline ===
start_time = time.time()
lcal = remove_baseline(lcal)
print("[INFO] Baseline removal time: {:.2f} s".format(time.time() - start_time))

# === After Baseline Removal ===
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(lcal[idx], vmin=0, vmax=1, cmap='magma')
plt.title("After Baseline Removal")
plt.colorbar()
plt.subplot(1, 2, 2)
plt.plot(lcal[idx].mean(axis=0), label='Row Mean (Post Correction)')
plt.title("Corrected Profile")
plt.legend()
plt.tight_layout()
plt.show()

# === Thresholding ===
print("[INFO] Applying thresholding...")
thresh_value = 0.5
max_value = 1
thresh_img = []

for i in range(len(lcal)):
    _, thresh_img1 = cv2.threshold(lcal[i], thresh_value, max_value, cv2.THRESH_BINARY)
    thresh_img.append(thresh_img1)

thresh_img = np.array(thresh_img)

# plt.figure(figsize=(5, 4))
# plt.imshow(thresh_img[0], vmin=0, vmax=1, cmap='magma')
# plt.colorbar()
# plt.title("Thresholded Image")
# plt.tight_layout()
# plt.show()

# === Morphology (Erosion & Dilation) ===
print("[INFO] Applying morphological operations...")
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
eroded_img = []
dilated_img = []

for i in tqdm(range(len(thresh_img))):
    eroded = cv2.erode(thresh_img[i], kernel, iterations=6)
    dilated = cv2.dilate(eroded, kernel, iterations=1)
    eroded_img.append(eroded)
    dilated_img.append(dilated)

eroded_img = np.array(eroded_img)
dilated_img = np.array(dilated_img)

# plt.figure(figsize=(5, 4))
# plt.imshow(eroded_img[0], vmin=0, vmax=1, cmap='magma')
# plt.colorbar()
# plt.title("Eroded Image")
# plt.tight_layout()
# plt.show()

# import imageio
# gif_path = 'dilated_img.gif'
# imageio.mimsave(gif_path, (dilated_img * 255).astype(np.uint8), duration=0.1)
# print(f"GIF 动画已保存到 {gif_path}")

# === Grid View of Dilated Images ===
# fig, axes = plt.subplots(nrows=6, ncols=6, figsize=(20, 18))
# for i, ax in enumerate(axes.flat):
#     ax.imshow(dilated_img[i], vmin=0, vmax=1, cmap='magma')
#     ax.set_title(f'Index {i}')
# plt.tight_layout()
# plt.show()

# === Connected Components and Contour Analysis ===
print("[INFO] Analyzing bead contours...")
ret, labels = cv2.connectedComponents(dilated_img[0].astype(np.uint8))
num_beads = ret - 1
bead_ids = np.arange(1, num_beads + 1)
print(f"[INFO] {num_beads} beads detected.")

contours, _ = cv2.findContours(dilated_img[0].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

bead_positions = []
for contour in contours:
    moments = cv2.moments(contour)
    cx = int(moments['m10'] / moments['m00'] + 1e-5)
    cy = int(moments['m01'] / moments['m00'] + 1e-5)
    bead_positions.append([cy, cx])

# === PCA Line Fitting ===
cov = np.cov(np.array(bead_positions).T)
eigenvalues, eigenvectors = np.linalg.eig(cov)
line_dir = eigenvectors[:, np.argmax(eigenvalues)]
mean_pos = np.mean(bead_positions, axis=0)
x1, y1 = (mean_pos[1] - 1000 * line_dir[1], mean_pos[0] - 1000 * line_dir[0])
x2, y2 = (mean_pos[1] + 1000 * line_dir[1], mean_pos[0] + 1000 * line_dir[0])

# === Project Each Bead Onto Line ===
bead_intersections = []
a = np.array([x1, y1])
b = np.array([x2, y2])
ab = b - a

for pos in bead_positions:
    p = np.array(pos)
    ap = p - a
    proj = a + np.dot(ap, ab) / np.dot(ab, ab) * ab
    if np.linalg.norm(p - proj) < 10:
        bead_intersections.append(proj)
    else:
        bead_intersections.append(p)

bead_intersections = np.array(bead_intersections)
bead_ids = np.arange(1, len(bead_positions) + 1)

print(f"[INFO] Bead positions (Y, X): {bead_positions}")
print(f"[INFO] Bead intersections:\n {bead_intersections}")

# === Visualization ===
contour_img = cv2.cvtColor(dilated_img[0], cv2.COLOR_GRAY2BGR)
cv2.drawContours(contour_img, contours, -1, (0, 0, 1), 2)
for i, pos in enumerate(bead_positions):
    cv2.circle(contour_img, tuple(pos[::-1]), 10, (0, 0, 1), -1)
    cv2.circle(contour_img, tuple(bead_intersections[i, ::-1].astype(int)), 10, (0, 1, 0), -1)

plt.figure()
plt.imshow(contour_img)
plt.title("Detected Beads and Intersections")
plt.tight_layout()
plt.show()

# === Summary Plot ===
bp = np.array(bead_positions)
plt.figure()
plt.plot(bp[:, 0], '.', label='Y positions')
plt.title("Bead Y-coordinates")
plt.xlabel("Bead Index")
plt.ylabel("Y Position")
plt.legend()
plt.tight_layout()
plt.show()

# Sensitivity threshold for detecting beads
sensitivity = 0.7

# Find the contours and intersections of the beads in each projection
bead_positions_all = []
bead_intersections_all = []
bead_ids_all = []
for i in range(dilated_img.shape[0]):
    # Find the contours of the beads in the current projection
    contours, _ = cv2.findContours(dilated_img[i].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Compute the center of mass of each contour using its moments
    bead_positions = []
    for j, contour in enumerate(contours):
        moments = cv2.moments(contour)
        if moments['m00'] == 0:
            continue
        cx = int(moments['m10'] / moments['m00'] + 1e-5)
        cy = int(moments['m01'] / moments['m00'] + 1e-5)
        bead_positions.append([cy, cx])

    # Fit a line to the bead positions using PCA
    cov = np.cov(np.array(bead_positions).T)
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    line_dir = eigenvectors[:, np.argmax(eigenvalues)]
    mean_pos = np.mean(bead_positions, axis=0)
    x1 = int(mean_pos[1] - 1000 * line_dir[1])
    y1 = int(mean_pos[0] - 1000 * line_dir[0])
    x2 = int(mean_pos[1] + 1000 * line_dir[1])
    y2 = int(mean_pos[0] + 1000 * line_dir[0])

    # Compute the intersection of each bead with the line
    bead_intersections = []
    bead_ids = []
    for j, bead_pos in enumerate(bead_positions):
        # Compute the projection of the bead onto the line
        a = np.array([x1, y1])
        b = np.array([x2, y2])
        p = np.array(bead_pos)
        ap = p - a
        ab = b - a
        proj = a + np.dot(ap, ab) / np.dot(ab, ab) * ab

        # Compute the distance from the projection to the bead
        dist = np.linalg.norm(p - proj)

        # If the distance is small enough, use the projection as the intersection point
        if dist < sensitivity:
            bead_intersections.append(proj)
        else:
            # Otherwise, use the bead position as the intersection point
            bead_intersections.append(bead_pos)

        # Append the bead ID to the list
        bead_ids.append(j + 1)

    bead_positions_all.append(bead_positions)
    bead_intersections_all.append(bead_intersections)
    bead_ids_all.append(bead_ids)

    # Assign unique IDs to each bead
    num_beads = len(bead_positions)
    bead_ids = np.arange(1, num_beads + 1)

    print(f"Detected {num_beads} beads with IDs {bead_ids}")
    print(f"Bead positions: {bead_positions}")
    print(f"Bead intersections: {bead_intersections}")

    # Show the result for some specific projections
    # if i in [0, 9, 19, 29]:
    #     fig, ax = plt.subplots()
    #     ax.imshow(contour_img)
    #     ax.set_title(f"Projection {i + 1}")
    #     ax.set_xlabel('X position')
    #     ax.set_ylabel('Y position')
    #     ax.set_aspect('equal', 'box')
    #     plt.show()

# Find unique lengths of the sublists
unique_lengths = set(len(sublist) for sublist in bead_positions_all)

print(f"Unique lengths of sublists: {unique_lengths}")

# Convert to numpy arrays with integer type
bead_ids_all = [np.array(item, dtype=int) for item in bead_ids_all]

# Find the maximum length
max_len_ids = max(len(item) for item in bead_ids_all)

# Pad the arrays
bead_ids_all_padded = np.array([
    np.pad(item, (0, max_len_ids - len(item)), 'constant', constant_values=-1)
    # Using -1 as the padding value for missing IDs
    for item in bead_ids_all
])

print("bead_ids_all_padded shape:", bead_ids_all_padded.shape)

# Ensure all items are numpy arrays of the same data type before padding
bead_positions_all = [np.array(item, dtype=float) if not isinstance(item, np.ndarray) else item.astype(float)
                      for item in bead_positions_all]

max_len = max(len(item) for item in bead_positions_all)

# Now, pad the arrays ensuring they all have the same length
bead_positions_all_padded = np.array(
    [np.pad(item, ((0, max_len - len(item)), (0, 0)), 'constant', constant_values=np.nan)
     for item in bead_positions_all])
# Find the maximum dimensions in each direction
max_len_0 = max(item.shape[0] for item in bead_positions_all)  # Number of beads
max_len_1 = max(item.shape[1] for item in bead_positions_all)  # Dimensions per bead, e.g., (x, y)

# Pad the arrays to these maximum dimensions
bead_positions_all_padded = np.array([
    np.pad(item, ((0, max_len_0 - item.shape[0]), (0, max_len_1 - item.shape[1])), 'constant', constant_values=np.nan)
    for item in bead_positions_all
])
print("bead_positions_all_padded shape:", bead_positions_all_padded.shape)

# Ensure all items are numpy arrays of the same data type before padding
bead_intersections_all = [np.array(item, dtype=float) if not isinstance(item, np.ndarray) else item.astype(float)
                          for item in bead_intersections_all]

# Find the maximum dimensions in each direction
max_len_intersections = max(len(item) for item in bead_intersections_all)
max_len_0_intersections = max(
    item.shape[0] for item in bead_intersections_all if item.ndim > 1)  # Number of intersections
max_len_1_intersections = 2  # Assuming intersections are 2D points (x, y)

# Pad the arrays to these maximum dimensions
bead_intersections_all_padded = np.array([
    np.pad(item, ((0, max_len_0_intersections - item.shape[0]), (0, max_len_1_intersections - item.shape[1])),
           'constant', constant_values=np.nan)
    if item.ndim > 1 else np.pad(item, (0, max_len_intersections - len(item)), 'constant', constant_values=np.nan)
    for item in bead_intersections_all
])

print("bead_intersections_all_padded shape:", bead_intersections_all_padded.shape)

# Convert to arrays
bead_positions_all = np.array(bead_positions_all_padded)
bead_intersections_all = np.array(bead_intersections_all_padded)
bead_ids_all = np.array(bead_ids_all_padded)

dilated_img = dilated_img.astype(np.uint8)

# Define the initial feature points as the bead positions in the first projection, along with their IDs
feature_points = np.hstack((bead_positions_all[0], np.array(bead_ids_all[0]).reshape(-1, 1))).astype(
    np.float32).reshape(-1, 1, 3)

# Define the Lucas-Kanade parameters
lk_params = dict(winSize=(20, 20),
                 maxLevel=5,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))

# Loop over each subsequent projection and track the feature points
trajectories = {}
for i in range(1, dilated_img.shape[0]):
    # Calculate the optical flow of the feature points from the previous frame to the current frame
    p1, st, err = cv2.calcOpticalFlowPyrLK(dilated_img[i - 1], dilated_img[i], feature_points[:, :, :2], None,
                                           **lk_params)

    # Remove any feature points that were not tracked successfully
    st = st.reshape(-1)
    feature_points = feature_points[st == 1]
    p1 = p1[st == 1]

    # Update the IDs of the feature points
    if feature_points.size > 0:
        feature_points[:, :, 2] = feature_points[:, :, 2] * st.reshape(-1, 1)

    # Detect new feature points in the current frame
    new_features = np.hstack((bead_positions_all[i], np.array(bead_ids_all[i]).reshape(-1, 1))).astype(
        np.float32).reshape(-1, 1, 3)
    feature_points = np.vstack((feature_points, new_features))

    # Calculate the trajectories of the feature points up to the current frame
    for j in range(feature_points.shape[0]):
        x, y, ID = feature_points[j, 0]
        if ID not in trajectories:
            trajectories[ID] = []
        trajectories[ID].append([y, x])

# Convert the trajectories to NumPy arrays
for ID in trajectories:
    trajectories[ID] = np.array(trajectories[ID])

# Fit ellipses to each bead trajectory up to the current frame
for ID in trajectories:
    if len(trajectories[ID]) > 5:  # Only fit ellipse if at least 6 points have been tracked
        # Convert the trajectory to a NumPy array
        trajectory = np.array(trajectories[ID])

        # Fit an ellipse to the trajectory using the method of moments
        ellipse = cv2.fitEllipse(trajectory)

        # Add the ellipse parameters to the trajectory dictionary
        trajectories[ID] = (trajectory, ellipse)
    else:
        # If there are not enough points, add None to the trajectory dictionary
        trajectories[ID] = (trajectory, None)

# Create a figure
fig, ax = plt.subplots(figsize=(15, 6.5))

# Create a colormap
cmap = plt.get_cmap('tab20')
colors = cmap(np.linspace(0, 1, len(trajectories)))

# Loop through the trajectories and plot each one with a unique color from the colormap
for index, (ID, data) in enumerate(trajectories.items()):
    if data[1] is not None:
        x, y = data[0][:, 0], data[0][:, 1]
        ax.plot(x, y, color=colors[index])  # , label=f"Bead {ID}")

# Set aspects and labels
ax.set_aspect('equal', 'box')
ax.set_xlabel('X position')
ax.set_ylabel('Y position')
ax.set_title('Bead Trajectories')
# Optionally, you can enable the legend if you want to identify the beads by color
# plt.legend(title="Bead ID", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

# Convert the remaining bead positions to NumPy arrays
bead_positions_all = [np.array(bead_positions) for bead_positions in bead_positions_all]

# Create a figure
fig, ax = plt.subplots(figsize=(15, 6.5))

# Create a colormap
cmap = plt.get_cmap('RdYlBu_r')
colors = cmap(np.linspace(0, 1, len(trajectories)))

# Plot bead positions with colors from the plasma color map
for i in range(dilated_img.shape[0]):
    if len(bead_positions_all[i]) > 0:
        x, y = bead_positions_all[i][:, 1], bead_positions_all[i][:, 0]
        ax.scatter(x, y, color=colors[i % len(colors)])

# Plot trajectories with matching colors
for index, (ID, data) in enumerate(trajectories.items()):
    if data[1] is not None:
        x, y = data[0][:, 0], data[0][:, 1]
        ax.plot(x, y, color=colors[index % len(colors)])  # , label=f"Bead {ID}")

# Set aspects and labels
ax.set_aspect('equal', 'box')
ax.set_xlabel('X position')
ax.set_ylabel('Y position')
ax.set_title('Bead Positions and Trajectories')
# Optionally, enable the legend to identify the beads by color
# plt.legend(title="Bead ID", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

fig, ax = plt.subplots(figsize=(5, 6))

ax.set_xlim(-200, 1792)
ax.set_ylim(-200, 2176)
for ID in trajectories:
    print(ID)
    if trajectories[ID][1] is not None:
        # Plot the trajectory and ellipse
        trajectory = trajectories[ID][0]
        plt.plot(trajectory[:, 0], trajectory[:, 1], 'skyblue')
    if trajectories[ID][1] is not None:
        # Plot the trajectory and ellipse
        ellipse = trajectories[ID][1]
        print(ellipse)
        if ellipse[1][0] < ellipse[1][1]:
            width = ellipse[1][1]
            height = ellipse[1][0]
        else:
            width = ellipse[1][0]
            height = ellipse[1][1]

        ellipse_patch = Ellipse(xy=ellipse[0], width=width, height=height, angle=ellipse[2], edgecolor='red',
                                facecolor='none')
        print(ellipse_patch)
        ax.add_patch(ellipse_patch)
# plt.axis('equal')
# plt.show()

fig, ax = plt.subplots(figsize=(5, 6))
for ID in trajectories:
    if trajectories[ID][1] is not None:
        # Plot the trajectory and ellipse
        trajectory = trajectories[ID][0]
        ellipse = trajectories[ID][1]
        plt.plot(trajectory[:, 0], trajectory[:, 1], 'skyblue')
        ellipse_patch = Ellipse(xy=ellipse[0], width=ellipse[1][0], height=ellipse[1][1], angle=ellipse[2],
                                edgecolor='red', facecolor='none')
        ax.add_patch(ellipse_patch)
plt.axis('equal')
plt.show()

# Calculate the major axis lengths of the ellipses
major_axis_lengths = [ellipse[1][1] for ID, (trajectory, ellipse) in trajectories.items() if ellipse is not None]

# Calculate the median major axis length
median_major_axis_length = np.median(major_axis_lengths)

# Define a threshold for major axis length as 2 times the median
major_axis_length_threshold = 1.6 * median_major_axis_length

# Remove outlier ellipses with major axis length greater than the threshold
for ID, (trajectory, ellipse) in list(trajectories.items()):
    if ellipse is not None and ellipse[1][1] > major_axis_length_threshold:
        trajectories.pop(ID)

# repeated code from above, make a function instead
# something is broken here and above
fig, ax = plt.subplots(1, figsize=(5, 6))

for ID in trajectories:
    if trajectories[ID][1] is not None:
        # Plot the trajectory and ellipse
        trajectory = trajectories[ID][0]
        ellipse = trajectories[ID][1]
        plt.plot(trajectory[:, 0], trajectory[:, 1], 'skyblue')
        ellipse_patch = Ellipse(xy=ellipse[0], width=ellipse[1][0], height=ellipse[1][1], angle=ellipse[2],
                                edgecolor='red', facecolor='none')
        plt.gca().add_patch(ellipse_patch)
plt.axis('equal')
plt.show()

trajectories_x = []
trajectories_y = []

for ID in trajectories:
    if trajectories[ID][1] is not None:
        trajectories_x.append(np.ravel(trajectories[ID][0][:, 0]).tolist())
        trajectories_y.append(np.ravel(trajectories[ID][0][:, 1]).tolist())

# Determine the maximum trajectory length
max_trajectory_length = max(len(x) for x in trajectories_x)

# Pad trajectories so they all have the same length
trajectories_x = np.array([np.pad(x, (0, max_trajectory_length - len(x)), 'constant', constant_values=np.nan)
                           for x in trajectories_x])
trajectories_y = np.array([np.pad(y, (0, max_trajectory_length - len(y)), 'constant', constant_values=np.nan)
                           for y in trajectories_y])

print("Padded trajectories_x shape:", trajectories_x.shape)
print("Padded trajectories_y shape:", trajectories_y.shape)

plt.scatter(trajectories_x[10], trajectories_y[10])


# fitting an ellipse
# based on Fitzgibbon, Pilu, & Fisher (1996) and Halir & Flusser (1998)
def fit(x, y):
    D1 = np.vstack([x ** 2, x * y, y ** 2]).T
    D2 = np.vstack([x, y, np.ones_like(x)]).T
    S1, S2, S3 = D1.T @ D1, D1.T @ D2, D2.T @ D2
    C1 = np.array([[0, 0, 2], [0, -1, 0], [2, 0, 0]])
    M = np.linalg.inv(C1) @ (S1 - S2 @ np.linalg.inv(S3) @ S2.T)
    vec = np.linalg.eig(M)[1]
    cond = 4 * (vec[0] * vec[2]) - vec[1] ** 2
    a1 = vec[:, np.nonzero(cond > 0)[0]]
    return np.vstack([a1, np.linalg.inv(-S3) @ S2.T @ a1]).flatten()


# estimating the errors using chi-squared
from uncertainties import ufloat


def errors(x, y, coeffs):
    z = np.vstack((x ** 2, x * y, y ** 2, x, y, np.ones_like(x)))
    numerator = np.sum(((coeffs.reshape(1, 6) @ z) - 1) ** 2)
    denominator = (len(x) - 6) * np.sum(z ** 2, axis=1)
    unc = np.sqrt(numerator / denominator)
    return tuple(ufloat(i, j) for i, j in zip(coeffs, unc))


# converting the coefficients to ellipse parameters
from uncertainties import umath


def convert(coeffs):
    a, b, c, d, e, f = coeffs
    b /= 2
    d /= 2
    e /= 2
    x0 = (c * d - b * e) / (b ** 2 - a * c)
    y0 = (a * e - b * d) / (b ** 2 - a * c)
    center = (x0, y0)
    numerator = 2 * (a * e ** 2 + c * d ** 2 + f * b ** 2 - 2 * b * d * e - a * c * f)
    denominator1 = (b * b - a * c) * ((c - a) * umath.sqrt(1 + 4 * b * b / ((a - c) * (a - c))) - (c + a))
    denominator2 = (b * b - a * c) * ((a - c) * umath.sqrt(1 + 4 * b * b / ((a - c) * (a - c))) - (c + a))
    major = umath.sqrt(numerator / denominator1) if numerator / denominator1 > 0 else 0
    minor = umath.sqrt(numerator / denominator2) if numerator / denominator2 > 0 else 0
    phi = .5 * umath.atan((2 * b) / (a - c))
    major, minor, phi = (major, minor, phi) if major > minor else (minor, major, np.pi / 2 + phi)
    return center, major, minor, phi


# generating fit ellipse line
def line(coeffs, n=100):
    t = np.linspace(0, 2 * np.pi, n)
    center, major, minor, phi = convert(coeffs)
    x = major * np.cos(t) * np.cos(phi) - minor * np.sin(t) * np.sin(phi) + center[0]
    y = major * np.cos(t) * np.sin(phi) + minor * np.sin(t) * np.cos(phi) + center[1]
    return x, y


# alternative using matplotlib artists
def artist(coeffs, *args, **kwargs):
    center, major, minor, phi = convert(coeffs)
    return Ellipse(xy=(center[0], center[1]), width=2 * major, height=2 * minor,
                   angle=np.rad2deg(phi), *args, **kwargs)


# obtaining the confidence interval/area for ellipse fit
def confidence_area(x, y, coeffs, f=1):  # f here is the multiple of sigma
    c = coeffs
    res = c[0] * x ** 2 + c[1] * x * y + c[2] * y ** 2 + c[3] * x + c[4] * y + c[5]
    sigma = np.std(res)
    #     print('Sigma = ', sigma)
    c_up = np.array([c[0], c[1], c[2], c[3], c[4], c[5] + f * sigma])
    c_do = np.array([c[0], c[1], c[2], c[3], c[4], c[5] - f * sigma])

    if convert(c_do) > convert(c_up):
        c_do, c_up = c_up, c_do
    return c_up, c_do


def fit(x, y):
    # Convert x and y to float type if they are not already
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    D1 = np.vstack([x ** 2, x * y, y ** 2]).T
    D2 = np.vstack([x, y, np.ones_like(x)]).T
    S1 = D1.T @ D1
    S2 = D1.T @ D2
    S3 = D2.T @ D2
    C1 = np.array([[0, 0, 2], [0, -1, 0], [2, 0, 0]], dtype=float)

    # Attempt to invert S3 with safe handling for singular matrices
    try:
        S3_inv = np.linalg.inv(S3)
    except np.linalg.LinAlgError:
        print("Warning: S3 is singular, using pseudo-inverse.")
        S3_inv = np.linalg.pinv(S3)

    M = np.linalg.inv(C1) @ (S1 - S2 @ S3_inv @ S2.T)

    # Eigenvalue decomposition
    eig_vals, eig_vecs = np.linalg.eig(M)
    cond = 4 * eig_vecs[0, :] * eig_vecs[2, :] - eig_vecs[1, :] ** 2
    valid_idx = np.where(cond > 0)[0]

    if valid_idx.size == 0:
        print("No valid ellipse found.")
        return None

    a1 = eig_vecs[:, valid_idx]
    return np.vstack([a1, np.linalg.solve(-S3, S2.T @ a1)]).flatten()


def fit(x, y):
    # Convert x and y to float type if they are not already
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    D1 = np.vstack([x ** 2, x * y, y ** 2]).T
    D2 = np.vstack([x, y, np.ones_like(x)]).T
    S1 = D1.T @ D1
    S2 = D1.T @ D2
    S3 = D2.T @ D2
    C1 = np.array([[0, 0, 2], [0, -1, 0], [2, 0, 0]], dtype=float)

    # Attempt to invert S3 with safe handling for singular matrices
    try:
        S3_inv = np.linalg.inv(S3)
    except np.linalg.LinAlgError:
        print("Warning: S3 is singular, using pseudo-inverse.")
        S3_inv = np.linalg.pinv(S3)

    M = np.linalg.inv(C1) @ (S1 - S2 @ S3_inv @ S2.T)

    # Eigenvalue decomposition
    eig_vals, eig_vecs = np.linalg.eig(M)
    cond = 4 * eig_vecs[0, :] * eig_vecs[2, :] - eig_vecs[1, :] ** 2
    valid_idx = np.where(cond > 0)[0]

    if valid_idx.size == 0:
        print("No valid ellipse found.")
        return None

    a1 = eig_vecs[:, valid_idx]
    return np.vstack([a1, np.linalg.solve(-S3, S2.T @ a1)]).flatten()


def errors(x, y, coeffs):
    z = np.vstack((x ** 2, x * y, y ** 2, x, y, np.ones_like(x)))
    if coeffs.size == 12:
        coeffs = coeffs.reshape(2, 6)
    elif coeffs.size == 6:
        coeffs = coeffs.reshape(1, 6)
    numerator = np.sum(((coeffs @ z) - 1) ** 2)
    denominator = (len(x) - 6) * np.sum(z ** 2, axis=1)
    unc = np.sqrt(numerator / denominator)
    return tuple(ufloat(i, j) for i, j in zip(coeffs.flatten(), unc))


def preprocess_data(x, y):
    """Preprocess the data by removing or imputing NaNs and Infs."""
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


N = len(trajectories)
# Assuming trajectories_x and trajectories_y are lists of arrays
for iii, (x, y) in enumerate(zip(trajectories_x, trajectories_y)):
    print(f'Processing ID = {iii}')

    # Preprocess the data to remove NaNs and Infs
    x, y = preprocess_data(np.array(x), np.array(y))

    # Check if there's enough data left to perform fitting
    if len(x) < 3 or len(y) < 3:
        print(f"Not enough valid data points to fit an ellipse for ID = {iii}")
        continue

    for ell_ii in range(N):
        try:
            # Perform the fitting and other calculations here
            params1 = fit(x, y)
            params1 = errors(x, y, params1)
            center, a, b, phi = convert(params1)
            # Additional logic and plotting...
        except np.linalg.LinAlgError as e:
            print(f"Linear algebra error for ID = {iii} during iteration {ell_ii}: {str(e)}")
            continue

# Assuming you have access to the original x, y data for ID 30
x_orig, y_orig = trajectories_x[1], trajectories_y[1]
x_clean, y_clean = preprocess_data(np.array(x_orig), np.array(y_orig))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.scatter(x_orig, y_orig, color='red', label='Original Data')
ax1.set_title('Original Data Points for ID 30')
ax1.set_xlabel('X Coordinate')
ax1.set_ylabel('Y Coordinate')
ax1.legend()

ax2.scatter(x_clean, y_clean, color='blue', label='Preprocessed Data')
ax2.set_title('Cleaned Data Points for ID 30')
ax2.set_xlabel('X Coordinate')
ax2.set_ylabel('Y Coordinate')
ax2.legend()

plt.show()

# Assume trajectories_x and trajectories_y are lists of arrays
x = np.array(trajectories_x, dtype=object)
y = np.array(trajectories_y, dtype=object)


def clean_coordinates(x, y):
    cleaned_x = []
    cleaned_y = []
    for xi, yi in zip(x, y):
        try:
            # Convert to float and replace non-convertible types with NaN
            xi = np.array(xi, dtype=float)
            yi = np.array(yi, dtype=float)
        except ValueError:
            # Handle the case where conversion to float fails
            continue

        if xi.size > 0 and yi.size > 0:  # Check if arrays are not empty
            data = np.column_stack((xi, yi))
            # Remove rows with NaN or Inf
            cleaned_data = data[~(np.isnan(data).any(axis=1) | np.isinf(data).any(axis=1))]
            cleaned_x.append(cleaned_data[:, 0])
            cleaned_y.append(cleaned_data[:, 1])
    return cleaned_x, cleaned_y


# Clean x and y
x_cleaned, y_cleaned = clean_coordinates(x, y)

# Print cleaned data (Example: print first cleaned trajectory if exists)
if x_cleaned and y_cleaned:
    print("Cleaned x[0]:", x_cleaned[0])
    print("Cleaned y[0]:", y_cleaned[0])

# number of iterations for outlier removal
N = 50

ell_all = []
err_all = []
Sigma_all = []
M = len(trajectories_x)
print('M = ', M)

for iii in range(M):
    print('ID = ', iii)

    sigma_ell = []
    # x= trajectories_x[iii]
    # y= trajectories_y[iii]
    x = np.array(x_cleaned[iii])
    y = np.array(y_cleaned[iii])

    for ell_ii in range(N):
        # do fitting
        params1 = fit(x, y)
        if params1 is None:
            continue
        params1 = errors(x, y, params1)

        center, a, b, phi = convert(params1)
        c_up, c_do = confidence_area(x, y, [i.n for i in params1], f=2)
        if convert(c_do) > convert(c_up):
            c_do, c_up = c_up, c_do

        # do the outlier removal
        mask1 = artist(c_up, ec='none')
        mask2 = artist(c_do, ec='none')

        # The ellipse1
        g_ell_center = getattr(mask1, "center")
        g_ell_width = getattr(mask1, "width")
        g_ell_height = getattr(mask1, "height")
        angle = getattr(mask1, "angle")

        g_ellipse = patches.Ellipse(g_ell_center, g_ell_width, g_ell_height, angle=angle, fill=False,
                                    edgecolor='skyblue', linewidth=2)
        ax.add_patch(g_ellipse)

        cos_angle = np.cos(np.radians(180. - angle))
        sin_angle = np.sin(np.radians(180. - angle))

        xc = x - g_ell_center[0]
        yc = y - g_ell_center[1]

        xct = xc * cos_angle - yc * sin_angle
        yct = xc * sin_angle + yc * cos_angle

        rad_cc = (xct ** 2 / (g_ell_width / 2.) ** 2) + (yct ** 2 / (g_ell_height / 2.) ** 2)

        # The ellipse2
        g_ell_center2 = getattr(mask2, "center")
        g_ell_width2 = getattr(mask2, "width")
        g_ell_height2 = getattr(mask2, "height")
        angle2 = getattr(mask2, "angle")

        g_ellipse2 = patches.Ellipse(g_ell_center2, g_ell_width2, g_ell_height2, angle=angle2, fill=False,
                                     edgecolor='skyblue', linewidth=2)
        ax.add_patch(g_ellipse2)

        cos_angle2 = np.cos(np.radians(180. - angle2))
        sin_angle2 = np.sin(np.radians(180. - angle2))

        xc = x - g_ell_center[0]
        yc = y - g_ell_center[1]

        xct = xc * cos_angle - yc * sin_angle
        yct = xc * sin_angle + yc * cos_angle

        rad_cc = (xct ** 2 / (g_ell_width / 2.) ** 2) + (yct ** 2 / (g_ell_height / 2.) ** 2)

        # The ellipse2
        g_ell_center2 = getattr(mask2, "center")
        g_ell_width2 = getattr(mask2, "width")
        g_ell_height2 = getattr(mask2, "height")
        angle2 = getattr(mask2, "angle")

        g_ellipse2 = patches.Ellipse(g_ell_center2, g_ell_width2, g_ell_height2, angle=angle2, fill=False,
                                     edgecolor='skyblue', linewidth=2)
        ax.add_patch(g_ellipse2)

        cos_angle2 = np.cos(np.radians(180. - angle2))
        sin_angle2 = np.sin(np.radians(180. - angle2))

        xc2 = x - g_ell_center2[0]
        yc2 = y - g_ell_center2[1]

        xct2 = xc2 * cos_angle2 - yc2 * sin_angle2
        yct2 = xc2 * sin_angle2 + yc2 * cos_angle2

        rad_cc2 = (xct2 ** 2 / (g_ell_width2 / 2.) ** 2) + (yct2 ** 2 / (g_ell_height2 / 2.) ** 2)

        # define new X and Y as modified
        xy_array = np.array([x, y])
        indices1 = np.where(rad_cc2 <= 1.)[0]
        if len(indices1) > 0:
            modified_array1 = np.delete(xy_array, indices1, 1)
            indices2 = np.intersect1d(np.where(rad_cc >= 1.)[0], np.arange(modified_array1.shape[1]))
            if len(indices2) > 0:
                modified_array2 = np.delete(modified_array1, indices2, 1)
                x = modified_array2[0]
                y = modified_array2[1]
            else:
                x = modified_array1[0]
                y = modified_array1[1]
        else:
            pass

        sigma = abs(((c_up - c_do) / 2)[5])
        sigma_ell.append(sigma)

    if params1 is None:
        continue
    ell = convert(params1)
    ell_all.append(ell)

    # # plot of fit and confidence area
    # fig, ax = plt.subplots()
    # ax.plot(x, y, 'sr', label='Data')
    # ax.plot(*line([i.n for i in params1]), '--b', lw=1, label='Fit')
    # ax.add_patch(artist(c_up, ec='none', fc='r', alpha=0.15, label=r'2$\sigma$'))
    # ax.add_patch(artist(c_do, ec='none', fc='white'))
    # ax.legend()
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # fig2, ax2 = plt.subplots()
    # ax2.plot(sigma_ell)
    # plt.title('sigma decay')
    #
    # print('last sigma value = ', sigma_ell[-1])
    # plt.show()

ell_all_array = np.array(ell_all, dtype=object)
M = len(ell_all_array)

# Collect ellipse parameters
import uncertainties as un

x_cen_nominal = np.array([un.nominal_value(ell_all_array[i][0][0]) for i in range(M)]).reshape(M, 1)
y_cen_nominal = np.array([un.nominal_value(ell_all_array[i][0][1]) for i in range(M)]).reshape(M, 1)
a_nominal_arr = np.array([un.nominal_value(ell_all_array[i][1]) for i in range(M)]).reshape(M, 1)
b_nominal_arr = np.array([un.nominal_value(ell_all_array[i][2]) for i in range(M)]).reshape(M, 1)
angles_nominal_arr = np.array([np.degrees(un.nominal_value(ell_all_array[i][3])) for i in range(M)]).reshape(M, 1)

# Plot all ellipses
fig, ax = plt.subplots(figsize=(10, 4))
for i in range(M):
    center = (x_cen_nominal[i][0], y_cen_nominal[i][0])
    width = 2 * a_nominal_arr[i][0]
    height = 2 * b_nominal_arr[i][0]
    angle = angles_nominal_arr[i][0]
    ellipse = patches.Ellipse(center, width, height, angle=angle, fill=False, edgecolor='skyblue')
    ax.add_patch(ellipse)
ax.set_aspect('equal')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Ellipses Plot')
ax.set_xlim(left=min(x_cen_nominal - 1.2 * a_nominal_arr), right=max(x_cen_nominal + 1.2 * a_nominal_arr))
ax.set_ylim(bottom=min(y_cen_nominal - 2.5 * b_nominal_arr), top=max(y_cen_nominal + 5 * b_nominal_arr))

plt.show()

# Calculate Z-scores for each parameter
z_x_center = zscore(x_cen_nominal)
z_y_center = zscore(y_cen_nominal)
z_major_axis = zscore(a_nominal_arr)
z_minor_axis = zscore(b_nominal_arr)

# Define threshold Z-score value
threshold = 1.6

# Find outliers indices for each parameter
outliers_x_center = np.where(np.abs(z_x_center) > threshold)[0]
outliers_y_center = np.where(np.abs(z_y_center) > threshold)[0]
outliers_major_axis = np.where(np.abs(z_major_axis) > threshold)[0]
outliers_minor_axis = np.where(np.abs(z_minor_axis) > threshold)[0]

# Combine indices of all outliers
all_outliers = np.unique(
    np.concatenate((outliers_x_center, outliers_y_center, outliers_major_axis, outliers_minor_axis)))

# Remove outliers from the ellipse parameters
x_cen_nominal_clean = np.delete(x_cen_nominal, all_outliers)
y_cen_nominal_clean = np.delete(y_cen_nominal, all_outliers)
a_nominal_arr_clean = np.delete(a_nominal_arr, all_outliers)
b_nominal_arr_clean = np.delete(b_nominal_arr, all_outliers)
angles_nominal_arr_clean = np.delete(angles_nominal_arr, all_outliers)

# Plot the remaining ellipses
fig, ax = plt.subplots(figsize=(10, 4))
for i in range(len(x_cen_nominal_clean)):
    center = (x_cen_nominal_clean[i], y_cen_nominal_clean[i])
    width = 2 * a_nominal_arr_clean[i]
    height = 2 * b_nominal_arr_clean[i]
    angle = angles_nominal_arr_clean[i]
    ellipse = patches.Ellipse(center, width, height, angle=angle, fill=False, edgecolor='skyblue')
    ax.add_patch(ellipse)
ax.set_aspect('equal')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Ellipses Plot (Outliers Removed)')
ax.set_xlim(left=min(x_cen_nominal - 1.2 * a_nominal_arr), right=max(x_cen_nominal + 1.2 * a_nominal_arr))
ax.set_ylim(bottom=min(y_cen_nominal - 2.5 * b_nominal_arr), top=max(y_cen_nominal + 5 * b_nominal_arr))
plt.show()

ell_all_array = np.array(ell_all, dtype=object)
M = ell_all_array.shape[0]
print('M:', M)
print(ell_all_array.shape)
print(len(ell_all_array))
print(ell_all_array[0][0][0])

x_cen_nominal_clean = np.delete(x_cen_nominal, all_outliers)
y_cen_nominal_clean = np.delete(y_cen_nominal, all_outliers)
a_nominal_arr_clean = np.delete(a_nominal_arr, all_outliers)
b_nominal_arr_clean = np.delete(b_nominal_arr, all_outliers)
angles_nominal_arr_clean = np.delete(angles_nominal_arr, all_outliers)

heights = y_cen_nominal_clean[:]
x_centers = x_cen_nominal_clean[:]
y_centers = y_cen_nominal_clean[:]
major_axes = a_nominal_arr_clean[:]
minor_axes = b_nominal_arr_clean[:]
angles = angles_nominal_arr_clean[:]

plt.plot(heights)
plt.plot(y_centers)
plt.plot(x_centers)
plt.plot(major_axes)
plt.plot(minor_axes)
plt.plot(angles)

# Calculate endpoints of line segments
line_points = []
for i in range(len(x_centers)):
    x0 = y_centers[i]
    y0 = x_centers[i]
    x1 = x0 + minor_axes[i] * np.cos((angles[i]))
    y1 = y0 + minor_axes[i] * np.sin((angles[i]))
    line_points.append([x0, y0, x1, y1])
line_points = np.array(line_points)

fig, ax = plt.subplots()

for x0, y0, x1, y1 in line_points:
    ax.plot([x0, x1], [y0, y1], color='skyblue')

# Set Y-axis limits
# ax.set_ylim(800, 990)  # Adjust the limits as needed

ax.set_title('Minor Axis, rotated 90 degrees')
ax.set_xlabel('heights')  # Label for the y-axis
ax.set_ylabel('X-centers')  # Label for the x-axis

plt.show()

fig, ax = plt.subplots()

for x0, y0, x1, y1 in line_points:
    ax.plot([x0, x1], [y0, y1], color='skyblue')

# Set Y-axis limits
ax.set_ylim(800, 990)  # Adjust the limits as needed

ax.set_title('Minor Axis, rotated 90 degrees')
ax.set_xlabel('heights')  # Label for the y-axis
ax.set_ylabel('X-centers')  # Label for the x-axis

plt.show()

# Calculate the length of each line segment
segment_lengths = np.sqrt((line_points[:, 2] - line_points[:, 0]) ** 2 + (line_points[:, 3] - line_points[:, 1]) ** 2)

# Find the 25th percentile of the length distribution
percentile_25 = scoreatpercentile(segment_lengths, 25)

# Filter out line segments in the 1st quarter of the length distribution
filtered_line_points = line_points[segment_lengths <= percentile_25]

# Plot the filtered line segments
fig, ax = plt.subplots()

for x0, y0, x1, y1 in filtered_line_points:
    ax.plot([x0, x1], [y0, y1], color='red')

ax.set_title('Filtered Line Segments')
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_ylim(880, 910)
plt.show()

# Global variables to hold the minimum point and value
global_min_point = None
global_min_value = None

# Prepare data for fitting
X = line_points[:, :2]  # Take the starting points of the lines
Y = minor_axes  # Corresponding minor axis lengths


# Higher-order model definition
def higher_order_model(xy, *coeffs):
    x, y = xy
    degree = 5  # Adjust based on the actual complexity of your model
    result = np.zeros_like(x)
    index = 0
    for i in range(degree + 1):
        for j in range(degree + 1 - i):
            result += coeffs[index] * (x ** i) * (y ** j)
            index += 1
    return result


# Prepare data for fitting
X = line_points[:, :2]  # Take the starting points of the lines
Y = minor_axes  # Corresponding minor axis lengths

# Define the initial guesses for the coefficients
degree = 5
num_coeffs = (degree + 1) * (degree + 2) // 2
initial_guess = np.zeros(num_coeffs)

# Fit the higher-order polynomial model
params, params_covariance = curve_fit(lambda xy, *params: higher_order_model(xy, *params), X.T, Y, p0=initial_guess)


# Objective function for minimization
def objective(xy):
    return higher_order_model(xy, *params)


# Set bounds for the optimization to realistic values
bounds = [(X[:, 0].min(), X[:, 0].max()), (X[:, 1].min(), X[:, 1].max())]

# Minimization to find the global minimum
result = minimize(objective, x0=[(X[:, 0].min() + X[:, 0].max()) / 2, (X[:, 1].min() + X[:, 1].max()) / 2],
                  bounds=bounds)
global_min_point = result.x
global_min_value = result.fun


def update_plot(elev=30, azim=30):
    global global_min_point, global_min_value
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Generate grid for surface
    x_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    y_range = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
    x_range, y_range = np.meshgrid(x_range, y_range)
    z_range = higher_order_model((x_range.ravel(), y_range.ravel()), *params).reshape(100, 100)

    ax.plot_surface(x_range, y_range, z_range, alpha=0.5, rstride=10, cstride=10, color='skyblue')
    ax.scatter(X[:, 0], X[:, 1], Y, color='c', s=50)  # Plot data points

    # Highlight the minimum point
    ax.scatter(result.x[0], result.x[1], result.fun, color='r', s=100, marker='*')

    # Set labels and title
    ax.set_title('Fitted Higher-Order Polynomial Surface with Minimum Point')
    ax.set_xlabel('X coordinates')
    ax.set_ylabel('Y coordinates')
    ax.set_zlabel('Minor Axis Lengths')

    # Set viewing angle
    ax.view_init(elev=elev, azim=azim)
    plt.show()


# Access global minimum point and value from outside
print("Accessed outside the function - Minimum point:", global_min_point)
print("Accessed outside the function - Minimum value:", global_min_value)
# Use interactive widget to adjust view angles
interactive_plot = interactive(update_plot, elev=(0, 90), azim=(0, 360))


def higher_order_model(xy, degree, *coeffs):
    x, y = xy
    result = np.zeros_like(x)
    index = 0
    for i in range(degree + 1):
        for j in range(degree + 1 - i):
            result += coeffs[index] * (x ** i) * (y ** j)
            index += 1
    return result


# Define the initial guesses for the coefficients
degree = 5
num_coeffs = (degree + 1) * (degree + 2) // 2
initial_guess = np.zeros(num_coeffs)

# Fit the higher-order polynomial model
params, params_covariance = curve_fit(lambda xy, *coeffs: higher_order_model(xy, degree, *coeffs), X.T, Y,
                                      p0=initial_guess)

# Example of using the fitted model (you can adapt this part as needed)
x_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
y_range = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
x_range, y_range = np.meshgrid(x_range, y_range)
z_range = higher_order_model((x_range.ravel(), y_range.ravel()), degree, *params).reshape(100, 100)

fig, ax = plt.subplots(figsize=(8, 6))
ax.contourf(x_range, y_range, z_range, levels=50, cmap='plasma')
ax.scatter(X[:, 0], X[:, 1], c='skyblue')
ax.set_title('Fitted Surface')
plt.show()

# Create a figure
fig, ax = plt.subplots(figsize=(3.5, 5))

# Generate a color map
cmap = plt.get_cmap('RdYlBu_r')

# Number of trajectories
num_trajectories = len(trajectories)
colors = cmap(np.linspace(0, 1, num_trajectories))

# Plot each trajectory with its color
for index, (ID, data) in enumerate(trajectories.items()):
    if data[1] is not None:
        x, y = data[0][:, 0], data[0][:, 1]
        ax.scatter(x, y, s=20, color=colors[index], marker='o', label=f'Bead {ID}')

# Scatter plot for global minimum point (assuming you have defined global_min_point)
ax.scatter(global_min_point[1], global_min_point[0], marker='*', s=300, color='red')

# Optional: If you want to label the global minimum point
ax.text(global_min_point[1], global_min_point[0], ' Piercing point', color='black', ha='right')

# Set labels and title
ax.set_xlabel('X position')
ax.set_ylabel('Y position')
ax.set_title('Bead Trajectories with Global Minimum')

# Show legend if you want to identify the beads by their ID
# plt.legend(title="Bead ID", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

fig, ax = plt.subplots(figsize=(3.5, 5))
ax.imshow(lcal[2], vmin=-0.03, vmax=0.4, cmap='gray')

# Generate a color map
cmap = plt.get_cmap('RdYlBu_r')

# Number of trajectories
num_trajectories = len(trajectories)
colors = cmap(np.linspace(0, 1, num_trajectories))

# Plot each trajectory with its color
for index, (ID, data) in enumerate(trajectories.items()):
    if data[1] is not None:
        x, y = data[0][:, 0], data[0][:, 1]
        ax.scatter(x, y, s=20, color=colors[index], marker='o', label=f'Bead {ID}')

# Scatter plot for global minimum point (assuming you have defined global_min_point)
ax.scatter(global_min_point[1], global_min_point[0], marker='*', s=300, color='red')

# Optional: If you want to label the global minimum point
ax.text(global_min_point[1], global_min_point[0], ' Global Min', color='white', ha='right')

# Set labels and title
ax.set_xlabel('X position')
ax.set_ylabel('Y position')
ax.set_title('Overlay of Bead Trajectories on Image')

# Show legend if you want to identify the beads by their ID
# plt.legend(title="Bead ID", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()
