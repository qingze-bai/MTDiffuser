# MTDiffuser

[[`Paper`](#)] [[`Dataset`](https://amos22.grand-challenge.org/)] [[`BibTeX`](#)]

![Variable-Shape design](assets/fig01.jpg?raw=true)

**MTDiffuser** is a medical image translation model that can handle various translation tasks based on prompt with just on trainning. We train MTDiffuser on 512×512 images from the [SynthRAD](https://synthrad2023.grand-challenge.org/) datasets and [Gold Atlas](https://aapm.onlinelibrary.wiley.com/doi/full/10.1002/mp.12748) datasets and Gold Atlas datasets. (a-b) MTDiff performs CT-to-MRI, MRI-to-CT, and CBCT-to-CT (from left to right) modality conversion in the head and pelvic region. (c) MTDiffuser also has anatomical consistency in the conversion of continuous slices.

<table>
    <tr>
        <td ><img src="assets/demo1.gif?raw=true"></td>
        <td ><img src="assets/demo2.gif?raw=true"></td>
    </tr>
    <tr>
        <td ><img src="assets/demo3.gif?raw=true"></td>
        <td ><img src="assets/demo4.gif?raw=true"></td>
    </tr>
</table>

## Installation

A suitable conda environment named ldm can be created and activated with:

```
conda env create -f environment.yml
conda activate mtdiff
```

## Data
The Diffuser model is trained on the collection of [SynthRAD](https://synthrad2023.grand-challenge.org/) datasets and [Gold Atlas](https://aapm.onlinelibrary.wiley.com/doi/full/10.1002/mp.12748) datasets. We have processed the data, such as resampling, cropping, etc. The processed data can be downloaded [here](https://amos22.grand-challenge.org/). We also have provided some processed data [here](https://amos22.grand-challenge.org/) for quick test.

After download and unzip the data to the user directory, you can specify the data directory in the  `models/autoencoder/autoencoder_kl_64x64x4.yaml` and `models/ldm/mtldm_v1_8_128.yaml`. Training and testing samples can be specified through files which can be find in `jsons/`

## Weights
We currently provide the following checkpoints:
- `autoencode_v1_kl_8.ckpt`: 120k steps at resolution `512x512` on [SynthRAD](https://synthrad2023.grand-challenge.org/) and [Gold Atlas](https://aapm.onlinelibrary.wiley.com/doi/full/10.1002/mp.12748).
- `mtldm-v1_8_128.ckpt`: 120k steps at resolution `512x512` on [SynthRAD](https://synthrad2023.grand-challenge.org/) and [Gold Atlas](https://aapm.onlinelibrary.wiley.com/doi/full/10.1002/mp.12748).


## Training and Inference




## Citation


