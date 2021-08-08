import random
import copy
from torch.utils.data.dataset import ConcatDataset
from torch.utils.data.sampler import Sampler


class RandomDomainSampler(Sampler):
    """Randomly sample :math:`N` domains, then randomly select :math:`K` samples in each domain to form a mini-batch of
    size :math:`N*K`.

    Args:
        data_source (ConcatDataset): dataset that contains data from multiple domains
        batch_size (int): mini-batch size (:math:`N*K` here)
        num_select_domains (int): number of domains to select in a single mini-batch (:math:`N` here)
    """

    def __init__(self, data_source: ConcatDataset, batch_size: int, num_select_domains: int):
        super(Sampler, self).__init__()
        self.num_all_domains = len(data_source.cumulative_sizes)
        self.num_select_domains = num_select_domains
        assert self.num_all_domains >= self.num_select_domains

        self.sample_idxes_per_domain = []
        start = 0
        for end in data_source.cumulative_sizes:
            idxes = [idx for idx in range(start, end)]
            self.sample_idxes_per_domain.append(idxes)
            start = end

        assert batch_size % num_select_domains == 0
        self.batch_size_per_domain = batch_size // num_select_domains
        self.length = len(list(self.__iter__()))

    def __iter__(self):
        sample_idxes_per_domain = copy.deepcopy(self.sample_idxes_per_domain)
        domain_idxes = [idx for idx in range(self.num_all_domains)]
        final_idxes = []
        stop_flag = False
        while not stop_flag:
            selected_domains = random.sample(domain_idxes, self.num_select_domains)

            for domain in selected_domains:
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


class RandomDomainMultiInstanceSampler(Sampler):

    def __init__(self, dataset, batch_size, num_select_domains, num_instances):
        super(Sampler, self).__init__()
        self.dataset = dataset
        self.sample_idxes_per_domain = {}
        for idx, (_, _, domain_id) in enumerate(self.dataset):
            if domain_id not in self.sample_idxes_per_domain:
                self.sample_idxes_per_domain[domain_id] = []
            self.sample_idxes_per_domain[domain_id].append(idx)
        self.num_all_domains = len(self.sample_idxes_per_domain)
        self.num_select_domains = num_select_domains
        assert self.num_all_domains >= self.num_select_domains

        assert batch_size % num_select_domains == 0
        self.batch_size_per_domain = batch_size // num_select_domains

        assert self.batch_size_per_domain % num_instances == 0
        self.num_instances = num_instances
        self.num_identities_per_domain = self.batch_size_per_domain // num_instances
        self.length = len(list(self.__iter__()))

    def __iter__(self):
        sample_idxes_per_domain = copy.deepcopy(self.sample_idxes_per_domain)
        domain_idxes = [idx for idx in range(self.num_all_domains)]
        final_idxes = []
        stop_flag = False
        while not stop_flag:
            selected_domains = random.sample(domain_idxes, self.num_select_domains)

            for domain in selected_domains:
                sample_idxes = sample_idxes_per_domain[domain]
                selected_idxes = self.sample_multi_instances(sample_idxes)
                final_idxes.extend(selected_idxes)

                for idx in selected_idxes:
                    sample_idxes_per_domain[domain].remove(idx)

                remaining_size = len(sample_idxes_per_domain[domain])
                if remaining_size < self.batch_size_per_domain:
                    stop_flag = True

        return iter(final_idxes)

    def sample_multi_instances(self, sample_idxes):
        idxes_per_identity = {}
        for idx in sample_idxes:
            _, pid, _ = self.dataset[idx]
            if pid not in idxes_per_identity:
                idxes_per_identity[pid] = []
            idxes_per_identity[pid].append(idx)

        pid_list = [pid for pid in idxes_per_identity if len(idxes_per_identity[pid]) >= self.num_instances]
        if len(pid_list) < self.num_identities_per_domain:
            return random.sample(sample_idxes, self.batch_size_per_domain)

        selected_idx = []
        selected_pid = random.sample(pid_list, self.num_identities_per_domain)
        for pid in selected_pid:
            selected_idx.extend(random.sample(idxes_per_identity[pid], self.num_instances))
        return selected_idx

    def __len__(self):
        return self.length
