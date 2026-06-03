from types import SimpleNamespace

from mmdet_comparison.datasets.density_batch_sampler import DensityAwareBatchSampler


class ListSampler:
    def __init__(self, indices, dataset):
        self.indices = list(indices)
        self.dataset = dataset

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def make_dataset(counts):
    data_list = [dict(instances=[{} for _ in range(count)]) for count in counts]
    return SimpleNamespace(data_list=data_list)


def test_density_sampler_splits_dense_images_from_large_batches():
    dataset = make_dataset([10, 20, 180, 30, 40, 220, 50])
    sampler = ListSampler(range(7), dataset)
    batch_sampler = DensityAwareBatchSampler(
        sampler,
        batch_size=3,
        dense_threshold=150,
        dense_batch_size=1,
    )

    assert list(batch_sampler) == [[0, 1], [2], [3, 4], [5], [6]]
