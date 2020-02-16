from datasets.office31 import Office31
from datasets.vision import VisionDataset

if __name__ == '__main__':
    print('starts!')
    office31 = Office31(root="data/office31", task='A', download=True)
    for file, target in office31:
        print(target)
        break

    vision = VisionDataset(root="/data/office/domain_adaptation_images/", data_list_file="/data/office/amazon_list.txt")
    for file, target in vision:
        print(target)
        break

