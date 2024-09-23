import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


def load_reconstruction_images(
    image_folder: str, num_timesteps: int, rollout_number: int = 1
) -> list[list[np.ndarray]]:
    true_imgs = []
    reconstructed_imgs = []
    imagined_imgs = []
    for i in range(num_timesteps):
        true_img = cv2.imread(
            os.path.join(
                image_folder, f"true_img_rollout_{rollout_number}_step_{i+1}.png"
            )
        )
        true_img = cv2.cvtColor(true_img, cv2.COLOR_BGR2RGB)
        reconstructed_img = cv2.imread(
            os.path.join(
                image_folder,
                f"reconstructed_img_rollout_{rollout_number}_step_{i+1}.png",
            )
        )
        reconstructed_img = cv2.cvtColor(reconstructed_img, cv2.COLOR_BGR2RGB)
        if i == 0:
            imagined_img = np.zeros_like(true_img)
        else:
            imagined_img = cv2.imread(
                os.path.join(
                    image_folder,
                    f"imagined_img_rollout_{rollout_number}_step_{i + 1}.png",
                )
            )
            imagined_img = cv2.cvtColor(imagined_img, cv2.COLOR_BGR2RGB)

        true_imgs.append(true_img)
        reconstructed_imgs.append(reconstructed_img)
        imagined_imgs.append(imagined_img)

    return [imagined_imgs, reconstructed_imgs, true_imgs]


def generate_image_reconstruction_plot(
    images: list[list[np.ndarray]], num_rows: int, num_cols: int
):

    assert len(images) == num_rows

    fig = plt.figure(figsize=(20, 5))
    gs = gridspec.GridSpec(num_rows, num_cols, wspace=0.1, hspace=0.1)

    row_titles = ["Imagined", "Reconstructed", "Actual"]

    for row in range(num_rows):
        for col in range(num_cols):
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(images[row][col], cmap="gray")
            ax.axis("off")

            # Adding row titles
            if col == 0:
                ax.text(
                    -0.5,
                    0.5,
                    row_titles[row],
                    va="center",
                    ha="right",
                    fontsize=14,
                    transform=ax.transAxes,
                )

            # Adding time labels
            if row == num_rows - 1:
                ax.annotate(
                    f"t={col+1}",
                    xy=(0.5, -0.35),
                    xycoords="axes fraction",
                    ha="center",
                    va="center",
                    fontsize=12,
                )

    plt.savefig("image_reconstruction_plot.png")


if __name__ == "__main__":
    image_folder = "/home/mattbarker/dev/somniloquy-dreamer-v3/world_model_evaluation/minigrid-occupancy-grid"
    num_timesteps = 16
    images = load_reconstruction_images(image_folder, num_timesteps)
    generate_image_reconstruction_plot(images, num_rows=3, num_cols=num_timesteps)
