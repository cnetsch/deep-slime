from PIL import Image
from pathlib import Path
import logging
import imageio
import numpy as np
from utils.dataset import BiofilmDataset

log = logging.getLogger()
log.setLevel(logging.INFO)


def read_tif(filename: Path) -> Image:
    return Image.open(str(filename))


def split_tif(img: Image):
    for page in range(img.n_frames):
        try:
            img.seek(page)
            yield img
        except EOFError:
            raise StopIteration


def tif_to_pngs(filename: Path, dir: Path = None, cnt_offset=0):
    filenames = []
    if dir is None:
        dir = filename.parent
    if not dir.exists():
        dir.mkdir(parents=True)
    img = read_tif(filename)
    for img_counter, img in enumerate(split_tif(img)):
        image_name = filename.stem.replace(BiofilmDataset.MASK_IDENTIFIER, "")
        image_name = image_name.replace("_untilt", "")
        dst = Path(dir, image_name + "_" + str(img_counter + cnt_offset) + ".png")
        img.save(dst)
        log.info(f"Saving frame to {dst}.")
        filenames.append(dst)
    return filenames


def tif_to_numpy(filename: Path):
    img = read_tif(filename)
    frames = [np.array(frame) for frame in split_tif(img)]
    return np.stack(frames)


def import_dataset(src: Path, dst: Path):
    masks_dir, imgs_dir = make_dataset_folder_structure(dst)
    for file in src.glob("*.tif"):
        save_to = masks_dir if is_mask(file) else imgs_dir
        tif_to_pngs(file, save_to)
    log.info(f"Succesfully finished import from {src}. Images saved to {dst}.")


def is_mask(file_name: Path):
    return BiofilmDataset.MASK_IDENTIFIER in file_name.stem


def make_dataset_folder_structure(dir):
    masks_dir, imgs_dir = Path(dir, "masks"), Path(dir, "imgs")
    for dir in [imgs_dir, masks_dir]:
        dir.mkdir(exist_ok=True, parents=True)
        with open(Path(dir, ".keep"), "w") as f:
            pass
    return masks_dir, imgs_dir


def write_tif_from_images(images: list, filename: Path):
    imageio.mimwrite(filename, images)
    log.info(f"Wrote {filename}.")


def write_tif_from_numpy(array: np.ndarray, filename: Path):
    images = [
        Image.fromarray(np.uint8(np.take(array, i, axis=0)))
        for i in range(array.shape[0])
    ]
    write_tif_from_images(images, filename)


if __name__ == "__main__":
    import shutil

    import_folder = Path(r"C:\Users\Christoph NETSCH\Desktop\repos\Pytorch-UNet\raw")
    dst_folder = Path(r"C:\Users\Christoph NETSCH\Desktop\repos\Pytorch-UNet\tmp")
    import_dataset(import_folder, dst_folder)
    sample = tif_to_numpy(
        r"C:\Users\Christoph NETSCH\Desktop\repos\Pytorch-UNet\raw\210501_Netsch_0078_Mode3D_untilt_binary_remove outliers.tif"
    )
    write_to = Path(dst_folder, r"reconstructed_sample.tif")
    write_tif_from_numpy(sample, write_to)
    shutil.rmtree(dst_folder)
