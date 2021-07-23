# Image-Adaptive-3DLUT
Learning Image-adaptive 3D Lookup Tables for High Performance Photo Enhancement in Real-time

this fork-branch is a modification of the [original repo](https://github.com/HuiZeng/Image-Adaptive-3DLUT) mainly for training and evaluation using PyTorch 1.X (tested on `1.9.0+cpu`).

## Downloads
for paper and original dataset, please refer to the author's original repo.

A model trained on the 480p resolution can be directly applied to images of 4K (or higher) resolution without performance drop. This can significantly speedup the training stage without loading the very heavy high-resolution images.

## Usage

### Requirements
Python3, requirements.txt

### Build
This `trilinear` extension is required inside the model and has to be built manually:
```bash
cd trilinear_cpp
sh setup.sh
```

### Training
#### paired training
     python3 image_adaptive_lut_train_paired.py --dataset_dir path/to/your/dataset/ --output_dir path/to/your/model_dir --n_cpu 3
Here are the key command line arguments:
* `--dataset_dir`: a directory containing sub-directories of `train/` and `test/` and within each of them, there are also sub-directories of `input/` and `output/`. As the author recommended, image size of about 480p is best (especially since the batch size is small and CUDA can't be used).
* `--n_cpu`: maximize this since training with CUDA is not possible (see this [issue](https://github.com/HuiZeng/Image-Adaptive-3DLUT/issues/40)) unless you are willing to run on pytorch `0.4.1` with CUDA 9.X
* `--batch_size`: only 1 sadly (see this [issue](https://github.com/HuiZeng/Image-Adaptive-3DLUT/issues/26))
* `--input_color_space`: only tested on the default `sRGB`
* `--output_dir`: check points, a `result.txt` and final models `LUTs_*.pth` and `classifier_*.pth`will be saved to this directory with the `_input_color_space`appended to the directory name

### Evaluation
    python3 demo_eval.py --image_dir path/to/your/image/directory/ --image_name input.jpg --output_dir path/to/output/directory --model_dir saved_models/your_model_sRGB/

## Citation
```
@article{zeng2020lut,
  title={Learning Image-adaptive 3D Lookup Tables for High Performance Photo Enhancement in Real-time},
  author={Zeng, Hui and Cai, Jianrui and Li, Lida and Cao, Zisheng and Zhang, Lei},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  volume={},
  number={},
  pages={},
  year={2020},
  publisher={IEEE}
}
```
