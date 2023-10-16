import torch
import numpy as np
import nibabel as nib
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from monai.data import load_decathlon_datalist
from PIL import Image

class Dataset_SynthRAD2023(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self._loaditem(index)

    def _loaditem(self, idx):
        filepath = self.data[idx]['image']
        item = np.load(filepath)

        target = item['target']
        edge = item['edge']
        source = item['source']
        prompt = item['prompt']
        filename = filepath.split('/')[-1]

        # Normalize target images to [-1, 1] as input
        target = (target - target.min()) / (target.max() - target.min()) * 2 - 1.0

        # Normalize source images to [-1, 1] as context
        source = (source - source.min()) / (source.max() - source.min()) * 2 - 1.0

        # Standardize edge map
        edge = edge.astype(np.float32)

        convert_type = filename.split('_')[-2]

        if convert_type == 'mr2ct':
            label = 0
        elif convert_type == 'ct2mr':
            label = 1
        elif convert_type == 'cbct2ct':
            label = 2
        return dict(target=target, edge=edge, source=source, txt=prompt, filename=filename, label=label)

def synth_collate(batch):
    targets = []
    edges = []
    sources = []
    txts = []
    filenames = []
    labels = []
    for item in batch:
        targets.append(item['target'])
        edges.append(item['edge'])
        sources.append(item['source'])
        txts.append(item['txt'])
        filenames.append(item['filename'])
        labels.append(item['label'])
    targets = torch.from_numpy(np.array(targets, np.float32))
    edges = torch.from_numpy(np.array(edges, np.float32))
    sources = torch.from_numpy(np.array(sources, np.float32))

    return dict(target=targets, edge=edges, source=sources, txt=txts, filename=filenames, label=labels)

def get_loader(config):
    base_dir = config['base_dir']
    json_list = config['json_list']

    train_files = []
    for json in json_list:
        files = load_decathlon_datalist(json, False, "training", base_dir=base_dir)
        train_files += files

    val_files = []
    for json in json_list:
        files = load_decathlon_datalist(json, False, "validation", base_dir=base_dir)
        val_files += files

    test_files = []
    for json in json_list:
        files = load_decathlon_datalist(json, False, "test", base_dir=base_dir)
        test_files += files

    print("Dataset all training: number of data: {}".format(len(train_files)))
    print("Dataset all validation: number of data: {}".format(len(val_files)))
    print("Dataset all test: number of data: {}".format(len(test_files)))

    train_ds = Dataset_SynthRAD2023(data=train_files)
    val_ds = Dataset_SynthRAD2023(data=val_files)
    test_ds = Dataset_SynthRAD2023(data=test_files)

    train_loader = DataLoader(train_ds,
                              batch_size=config["batch_size"],
                              shuffle=True,
                              num_workers=config.get("num_workers", 0),
                              pin_memory=config.get("pin_memory", False),
                              collate_fn=synth_collate
                              )

    val_loader = DataLoader(val_ds,
                            batch_size=config["batch_size"],
                            shuffle=False,
                            num_workers=config.get("num_workers", 0),
                            pin_memory=config.get("pin_memory", False),
                            collate_fn=synth_collate
                            )

    test_loader = DataLoader(test_ds,
                            batch_size=config["batch_size"],
                            shuffle=False,
                            num_workers=config.get("num_workers", 0),
                            pin_memory=config.get("pin_memory", False),
                            collate_fn=synth_collate
                            )

    return [train_loader, val_loader, test_loader]