import argparse
import logging
import os
import numpy as np
import sys
import torch
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import shutil
import torch.nn.functional as F
from unet import UNet
from utils.conversions import tif_to_pngs, write_tif_from_images
from utils.dataset import BiofilmDataset
from torch.utils.data import DataLoader

TMP_DIR = "tmp/"
OUT_DIR = "outputs/"

def predict(net, device, batch_size=1, img_scale=0.5, out_threshold=0.5, file_num=0, total_files=0):

    dataset = BiofilmDataset(TMP_DIR, scale=img_scale)
    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True
    )

    logging.info(
        f"""Starting predictions:
        Batch size:      {batch_size}
        Device:          {device.type}
        Images scaling:  {img_scale}
    """
    )

    net.train()
    batches = len(data_loader)
    batch_nr = 0

    with tqdm(total=batches, desc=f"File {file_num + 1}/{total_files} ", unit="frame") as pbar:
        for batch in data_loader:
            predict_batch(net, device, out_threshold, pbar, batch)
            batch_nr += 1


def predict_batch(net, device, out_threshold, pbar, batch):
    imgs = batch["image"]
    img_names = batch["name"]
    assert imgs.shape[1] == net.n_channels, (
                f"Network has been defined with {net.n_channels} input channels, "
                f"but loaded images have {imgs.shape[1]} channels. Please check that "
                "the images are loaded correctly."
            )

    imgs = imgs.to(device=device, dtype=torch.float32)
    output = net(imgs)

    pbar.set_postfix(**{"image": batch["name"]})
    pbar.update(imgs.shape[0])

    if net.n_classes > 1:
        probs = F.softmax(output, dim=1)
    else:
        probs = torch.sigmoid(output)

    mask = probs > out_threshold

    for frame_nr in range(mask.shape[0]):
        save_frame(img_names, mask, frame_nr)

def save_frame(img_names, mask, frame_nr):
    frame = mask[frame_nr, :, :, :]
    frame = frame.squeeze(0)
    mask_img = Image.fromarray(np.uint8(255*frame.numpy()))
    saveto = f"{TMP_DIR}{img_names[frame_nr]}_out.png"
    mask_img.save(saveto)

def get_args():
    parser = argparse.ArgumentParser(
        description="Predict masks from input images",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        "-m",
        default="models/best.pth",
        metavar="FILE",
        help="Specify the file in which the model is stored",
    )
    parser.add_argument(
        "--input",
        "-i",
        metavar="INPUT",
        nargs="+",
        help="filenames of input images",  
        required=True
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        metavar="B",
        type=int,
        nargs="?",
        default=1,
        help="Batch size",
        dest="batchsize",
    )
    # parser.add_argument(
    #     "--viz",
    #     "-v",
    #     action="store_true",
    #     help="Visualize the images as they are processed",
    #     default=False,
    # )
    # parser.add_argument(
    #     "--no-save",
    #     "-n",
    #     action="store_true",
    #     help="Do not save the output masks",
    #     default=False,
    # )
    parser.add_argument(
        "--mask-threshold",
        "-t",
        type=float,
        help="Minimum probability value to consider a mask pixel white",
        default=0.5,
    )
    parser.add_argument(
        "--scale",
        "-s",
        type=float,
        help="Scale factor for the input images",
        default=1.0,
    )

    return parser.parse_args()


if __name__ == "__main__":
    try:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
        args = get_args()

        if isinstance(args.input, list):
            input_tifs = [Path(i) for i in args.input]
        else:
            input_tifs = [Path(args.input)]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device {device}")

        # Change here to adapt to your data
        # n_channels=3 for RGB images
        # n_classes is the number of probabilities you want to get per pixel
        #   - For 1 class and background, use n_classes=1
        #   - For 2 classes, use n_classes=1
        #   - For N > 2 classes, use n_classes=N
        net = UNet(n_channels=1, n_classes=1, bilinear=True)
        logging.info(
            f"Network:\n"
            f"\t{net.n_channels} input channels\n"
            f"\t{net.n_classes} output channels (classes)\n"
            f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling'
        )

        for file_num, input_tif in enumerate(input_tifs):
            try:
                shutil.rmtree(TMP_DIR)
            except:
                pass
            tif_to_pngs(input_tif, Path(TMP_DIR))

            net.load_state_dict(torch.load(args.model, map_location=device))
            logging.info(f"Model loaded from {args.model}")

            net.to(device=device)
            # faster convolutions, but more memory
            # cudnn.benchmark = True

            predict(
                net=net, batch_size=args.batchsize, device=device, img_scale=args.scale, out_threshold=args.mask_threshold, file_num=file_num,
                total_files=len(input_tifs)
            )

            output_frames = sorted([f for f in Path(TMP_DIR).glob("*_out.png")])
            write_tif_from_images([Image.open(f) for f in output_frames], Path(OUT_DIR, f"{input_tif.stem}_binary.tif"))

    except KeyboardInterrupt:
        logging.info("Interrupted")
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
            
    finally:
        shutil.rmtree(TMP_DIR)

