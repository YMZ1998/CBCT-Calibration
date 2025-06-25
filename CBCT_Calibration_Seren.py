import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
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
fig, ax = plt.subplots(1, 3, figsize=(12, 4))
ax[0].imshow(dc, cmap='magma')
ax[0].set_title('Dark Current (DC)')
a1 = ax[1].imshow(ob, cmap='magma')
fig.colorbar(a1, ax=ax[1])
ax[1].set_title('Open Beam (OB)')
a2 = ax[2].imshow(proj[1], cmap='magma')
fig.colorbar(a2, ax=ax[2])
ax[2].set_title('Sample Projection')
plt.tight_layout()
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

plt.figure(figsize=(5, 4))
plt.imshow(thresh_img[0], vmin=0, vmax=1, cmap='magma')
plt.colorbar()
plt.title("Thresholded Image")
plt.tight_layout()
plt.show()

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

plt.figure(figsize=(5, 4))
plt.imshow(eroded_img[0], vmin=0, vmax=1, cmap='magma')
plt.colorbar()
plt.title("Eroded Image")
plt.tight_layout()
plt.show()

import imageio

gif_path = 'dilated_img.gif'
imageio.mimsave(gif_path, (dilated_img * 255).astype(np.uint8), duration=0.1)

print(f"GIF 动画已保存到 {gif_path}")

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
