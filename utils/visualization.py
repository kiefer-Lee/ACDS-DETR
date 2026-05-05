from pathlib import Path

import matplotlib.pyplot as plt


def save_query_points(image, points, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.imshow(image)
    plt.scatter(points[:, 0], points[:, 1], s=8, c="red")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

