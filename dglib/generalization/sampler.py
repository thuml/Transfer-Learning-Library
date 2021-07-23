import random
import copy
from torch.utils.data.dataset import ConcatDataset
from torch.utils.data.sampler import Sampler


class DefaultSampler(Sampler):
    """Traverse all N domains, for each domain random select K samples to form a mini-batch of size N*K

    Args:
        data_source (ConcatDataset): dataset that contains data from multiple domains
        batch_size (int): mini-batch size (N*K here)
    """

    def __init__(self, data_source: ConcatDataset, batch_size: int):
        super(Sampler, self).__init__()
        self.num_all_domains = len(data_source.cumulative_sizes)

        self.sample_idxes_per_domain = []
        start = 0
        for end in data_source.cumulative_sizes:
            idxes = [idx for idx in range(start, end)]
            self.sample_idxes_per_domain.append(idxes)
            start = end

        assert batch_size % self.num_all_domains == 0
        self.batch_size_per_domain = batch_size // self.num_all_domains
        self.length = len(list(self.__iter__()))

    def __iter__(self):
        sample_idxes_per_domain = copy.deepcopy(self.sample_idxes_per_domain)
        final_idxes = []
        stop_flag = False
        while not stop_flag:
            for domain in range(self.num_all_domains):
                sample_idxes = sample_idxes_per_domain[domain]
                selected_idxes = random.sample(sample_idxes, self.batch_size_per_domain)
                final_idxes.extend(selected_idxes)

                for idx in selected_idxes:
                    sample_idxes_per_domain[domain].remove(idx)

                remaining_size = len(sample_idxes_per_domain[domain])
                if remaining_size < self.batch_size_per_domain:
                    stop_flag = True

        return iter(final_idxes)

    def __len__(self):
        return self.length
