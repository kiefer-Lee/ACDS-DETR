import torch


def collate_fn(batch):
    images, targets = zip(*batch)
    max_h = max(img.shape[-2] for img in images)
    max_w = max(img.shape[-1] for img in images)
    batch_tensor = images[0].new_zeros((len(images), 3, max_h, max_w))
    mask = torch.ones((len(images), max_h, max_w), dtype=torch.bool)
    for i, img in enumerate(images):
        c, h, w = img.shape
        batch_tensor[i, :c, :h, :w] = img
        mask[i, :h, :w] = False
    return {"tensors": batch_tensor, "mask": mask}, list(targets)

