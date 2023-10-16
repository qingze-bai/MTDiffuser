import numpy as np
import nibabel as nib
import nibabel.processing as nip

def info_nib(img):
    affine, header= img.affine, img.header
    shape, spacing = img.get_fdata().shape, header.get_zooms()
    return affine, header, shape, spacing


def resample_nib(img, voxel_spacing=(1, 1, 1), order=3):
    """Resamples the nifti from its original spacing to another specified spacing

    Parameters:
    ----------
    img: nibabel image
    voxel_spacing: a tuple of 3 integers specifying the desired new spacing
    order: the order of interpolation

    Returns:
    ----------
    new_img: The resampled nibabel image

    """
    # resample to new voxel spacing based on the current x-y-z-orientation
    aff = img.affine
    shp = img.shape
    zms = img.header.get_zooms()
    # Calculate new shape
    new_shp = tuple(np.rint([
        shp[0] * zms[0] / voxel_spacing[0],
        shp[1] * zms[1] / voxel_spacing[1],
        shp[2] * zms[2] / voxel_spacing[2]
    ]).astype(int))
    new_aff = nib.affines.rescale_affine(aff, shp, voxel_spacing, new_shp)
    new_img = nip.resample_from_to(img, (new_shp, new_aff), mode='nearest', order=order, cval=-1024)
    # print("[*] Image resampled to voxel size:", voxel_spacing)
    return new_img

def resize_data(data, new_size=(512,512,512)):
    ih, iw, id = data.shape
    nh, nw, nd = new_size

    new_data = np.zeros(new_size, dtype=np.float32)
    if data.min() > -100:
        new_data[:, :, :] = 0
    else:
        new_data[:, :, :] = data.min()

    ih_start = iw_start = id_start = nh_start = nw_start = nd_start = 0
    ih_end = ih
    iw_end = iw
    id_end = id
    nh_end = nh
    nw_end = nw
    nd_end = nd

    if nh >= ih:
        nh_start = (nh - ih) // 2
        nh_end = nh_start + ih
    else:
        ih_start = (ih - nh) // 2
        ih_end = ih_start + nh

    if nw >= iw:
        nw_start = (nw - iw) // 2
        nw_end = nw_start + iw
    else:
        iw_start = (iw - nw) // 2
        iw_end = iw_start + nw

    if nd >= id:
        nd_start = (nd - id) // 2
        nd_end = nd_start + id
    else:
        id_start = (id - nd) // 2
        id_end = id_start + nd

    new_data[nh_start:nh_end, nw_start:nw_end, nd_start:nd_end] = data[ih_start:ih_end, iw_start:iw_end, id_start:id_end]
    return new_data