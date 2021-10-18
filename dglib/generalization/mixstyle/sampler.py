"""
@author: Baixu Chen
@contact: cbx_99_hasta@outlook.com
"""
import random
import copy
from torch.utils.data.dataset import ConcatDataset
from torch.utils.data.sampler import Sampler


class RandomDomainMultiInstanceSampler(Sampler):
    r"""Randomly sample :math:`N` domains, then randomly select :math:`P` instances in each domain, for each instance,
    randomly select :math:`K` images to form a mini-batch of size :math:`N\times P\times K`.

    Args:
        dataset (ConcatDataset): dataset that contains data from multiple domains
        batch_size (int): mini-batch size (:math:`N\times P\times K` here)
        n_domains_per_batch (int): number of domains to select in a single mini-batch (:math:`N` here)
        num_instances (int): number of instances to select in each domain (:math:`K` here)
    """

    def __init__(self, dataset, batch_size, n_domains_per_batch, num_instances):
        super(Sampler, self).__init__()
        self.dataset = dataset
        self.sample_idxes_per_domain = {}
        for idx, (_, _, domain_id) in enumerate(self.dataset):
            if domain_id not in self.sample_idxes_per_domain:
                self.sample_idxes_per_domain[domain_id] = []
            self.sample_idxes_per_domain[domain_id].append(idx)
        self.n_domains_in_dataset = len(self.sample_idxes_per_domain)
        self.n_domains_per_batch = n_domains_per_batch
        assert self.n_domains_in_dataset >= self.n_domains_per_batch

        assert batch_size % n_domains_per_batch == 0
        self.batch_size_per_domain = batch_size // n_domains_per_batch

        assert self.batch_size_per_domain % num_instances == 0
        self.num_instances = num_instances
        self.num_classes_per_domain = self.batch_size_per_domain // num_instances
        self.length = len(list(self.__iter__()))

    def __iter__(self):
        sample_idxes_per_domain = copy.deepcopy(self.sample_idxes_per_domain)
        domain_idxes = [idx for idx in range(self.n_domains_in_dataset)]
        final_idxes = []
        stop_flag = False
        while not stop_flag:
            selected_domains = random.sample(domain_idxes, self.n_domains_per_batch)

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
        idxes_per_cls = {}
        for idx in sample_idxes:
            _, cls, _ = self.dataset[idx]
            if cls not in idxes_per_cls:
                idxes_per_cls[cls] = []
            idxes_per_cls[cls].append(idx)

        cls_list = [cls for cls in idxes_per_cls if len(idxes_per_cls[cls]) >= self.num_instances]
        if len(cls_list) < self.num_classes_per_domain:
            return random.sample(sample_idxes, self.batch_size_per_domain)

        selected_idxes = []
        selected_classes = random.sample(cls_list, self.num_classes_per_domain)
        for cls in selected_classes:
            selected_idxes.extend(random.sample(idxes_per_cls[cls], self.num_instances))
        return selected_idxes

    def __len__(self):
        return self.length
