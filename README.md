# Deep-Slime: semantic segmentation for biofilms (forked from  PyTorch-UNet)

**Input**

[![xscode](https://img.shields.io/badge/Available%20on-xs%3Acode-blue?style=?style=plastic&logo=appveyor&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAMAAACdt4HsAAAAGXRFWHRTb2Z0d2FyZQBBZG9iZSBJbWFnZVJlYWR5ccllPAAAAAZQTFRF////////VXz1bAAAAAJ0Uk5T/wDltzBKAAAAlUlEQVR42uzXSwqAMAwE0Mn9L+3Ggtgkk35QwcnSJo9S+yGwM9DCooCbgn4YrJ4CIPUcQF7/XSBbx2TEz4sAZ2q1RAECBAiYBlCtvwN+KiYAlG7UDGj59MViT9hOwEqAhYCtAsUZvL6I6W8c2wcbd+LIWSCHSTeSAAECngN4xxIDSK9f4B9t377Wd7H5Nt7/Xz8eAgwAvesLRjYYPuUAAAAASUVORK5CYII=)](https://xscode.com/milesial/Pytorch-UNet)

![input image from test dataset](resources/example_in.png)

**Output**

![output image](resources/example_out.png)


Forked from implementation of the [U-Net](https://arxiv.org/abs/1505.04597) in PyTorch .

This model was trained from scratch with aprox. 18000 images (no data augmentation) and scored a [dice coefficient](https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient) of 0.97 on over 2000 test images. This score could be improved with more training, data augmentation, fine tuning, playing with CRF post-processing, and applying more weights on the edges of the masks.

## Set Up

Follow these steps to get started:

1. Install this repository from source:

   `git clone https://github.com/cnetsch/deep-slime.git`

   `cd deep-slime`

2. Install requirements:

   `pip install -r requirements.txt`

3. If you are not training a custom model, copy a pretrained model to `models/` (the default model is named `models/best.pth`, name accordingly unless passing a different model name to`predict.py` explicitly).

4. You can now run the code from the command line.

## Usage
**Note : Use Python 3.6 or newer**
### Prediction

After training your model and saving it to `models/best.pth`, you can easily test the output masks on your images via the CLI.

To predict a single image and save it:

`python predict.py -i image.jpg`

To predict a multiple images and show them without saving them:

`python predict.py -i image1.jpg image2.jpg`

Output images are stored under the corresponding filename of the input image in `outputs/`.

```shell script
> python predict.py -h
usage: predict.py [-h] [--model FILE] --input INPUT [INPUT ...]
                  [--output INPUT [INPUT ...]] [--viz] [--no-save]
                  [--mask-threshold MASK_THRESHOLD] [--scale SCALE]

Predict masks from input images

optional arguments:
  -h, --help            show this help message and exit
  --model FILE, -m FILE
                        Specify the file in which the model is stored
                        (default: MODEL.pth)
  --input INPUT [INPUT ...], -i INPUT [INPUT ...]
                        filenames of input images (default: None)
  --viz, -v             Visualize the images as they are processed (default:
                        False) (not implemented yet)
  --no-save, -n         Do not save the output masks (default: False) (not implemented yet)
  --mask-threshold MASK_THRESHOLD, -t MASK_THRESHOLD
                        Minimum probability value to consider a mask pixel
                        white (default: 0.5)
  --scale SCALE, -s SCALE
                        Scale factor for the input images (default: 0.5)
```
You can specify which model file to use with `--model models/my_model.pth`.

### Training

```shell script
> python train.py -h
usage: train.py [-h] [-e E] [-b [B]] [-l [LR]] [-f LOAD] [-s SCALE] [-v VAL]

Train the UNet on images and target masks

optional arguments:
  -h, --help            show this help message and exit
  -e E, --epochs E      Number of epochs (default: 5)
  -b [B], --batch-size [B]
                        Batch size (default: 1)
  -l [LR], --learning-rate [LR]
                        Learning rate (default: 0.1)
  -f LOAD, --load LOAD  Load model from a .pth file (default: False)
  -s SCALE, --scale SCALE
                        Downscaling factor of the images (default: 0.5)
  -v VAL, --validation VAL
                        Percent of the data that is used as validation (0-100)
                        (default: 15.0)

```
By default, the `scale` is 0.5, so if you wish to obtain better results (but use more memory), set it to 1.

The input images and target masks should be in the `data/imgs` and `data/masks` folders respectively.

## Tensorboard
You can visualize in real time the train and test losses, the weights and gradients, along with the model predictions with tensorboard:

`tensorboard --logdir=runs`

You can find a reference training run with the Caravana dataset on [TensorBoard.dev](https://tensorboard.dev/experiment/1m1Ql50MSJixCbG1m9EcDQ/#scalars&_smoothingWeight=0.6) (only scalars are shown currently).

## Notes on memory

The model has be trained from scratch on a GTX970M 3GB.
Predicting images of 1918*1280 takes 1.5GB of memory.
Training takes much approximately 3GB, so if you are a few MB shy of memory, consider turning off all graphical displays.
This assumes you use bilinear up-sampling, and not transposed convolution in the model.

## Support

Personalized support for issues with this repository, or integrating with your own dataset, available on [xs:code](https://xscode.com/milesial/Pytorch-UNet).


---

The algorithm and general implementation was forked from [Pytorch-UNet](https://github.com/milesial/Pytorch-UNet).

Original paper by Olaf Ronneberger, Philipp Fischer, Thomas Brox: [https://arxiv.org/abs/1505.04597](https://arxiv.org/abs/1505.04597)

![network architecture](https://i.imgur.com/jeDVpqF.png)
