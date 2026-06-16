# DeepJSCC Reproduction

This repository is a PyTorch reproduction of the original DeepJSCC paper:

> E. Bourtsoulatze, D. Burth Kurka, and D. Gündüz,  
> "Deep Joint Source-Channel Coding for Wireless Image Transmission,"  
> IEEE Transactions on Cognitive Communications and Networking, 2019.  
> Paper: https://arxiv.org/abs/1809.01733

This project can be used as a baseline for later experiments on deep joint source-channel coding.

## Model

The implementation follows the original CIFAR-10 DeepJSCC architecture.

### Encoder

The encoder is a CNN with PReLU activations:

```text
Conv 5x5, 16 channels, stride 2 -> PReLU
Conv 5x5, 32 channels, stride 2 -> PReLU
Conv 5x5, 32 channels, stride 1 -> PReLU
Conv 5x5, 32 channels, stride 1 -> PReLU
Conv 5x5, C  channels, stride 1 -> PReLU
Power normalization
```

For CIFAR-10, the bandwidth ratio is controlled by the last channel number `C`:

```text
k/n = C / 96
```

For example:

```text
k/n = 1/12 -> C = 8
k/n = 1/6  -> C = 16
```

### Channel

The communication channel is implemented as a non-trainable layer.

Supported channels:

```text
AWGN:                 z_hat = z + n
Slow Rayleigh fading: z_hat = h z + n
```

For slow Rayleigh fading, one channel coefficient `h` is sampled for each transmitted image and kept constant during the whole image transmission, following the original paper.

### Decoder

The decoder mirrors the encoder with transposed convolution layers:

```text
Transposed Conv 5x5, 32 channels, stride 1 -> PReLU
Transposed Conv 5x5, 32 channels, stride 1 -> PReLU
Transposed Conv 5x5, 32 channels, stride 1 -> PReLU
Transposed Conv 5x5, 16 channels, stride 2 -> PReLU
Transposed Conv 5x5, 3  channels, stride 2 -> Sigmoid
```

## Dataset

The default dataset is CIFAR-10, as used in the original paper.

```text
Training set: 50,000 CIFAR-10 training images
Test set:     10,000 CIFAR-10 test images
Image size:   32 x 32 RGB
```

## Training

Example: train an AWGN model with `k/n = 1/12` and training SNR = 7 dB.

```bash
python3 train.py \
  --download \
  --channel awgn \
  --ratio 0.08333333333333333 \
  --snr 7 \
  --steps 600000
```

## Evaluation

```bash
python3 evaluate.py \
  runs/cifar10_awgn_ratio-0.083333_snr-7_seed-0/checkpoint-600000.pt \
  --test-snrs 1 4 7 10 13 16 19 22 25 \
  --repeats 10 \
  --output results/ratio-1-12_train-snr-7.csv
```

## Reproduce CIFAR-10 AWGN Curves

Train all models used for the CIFAR-10 AWGN SNR sweep:

```bash
python3 scripts/run_awgn_sweep.py --download --steps 600000
```

Evaluate all checkpoints:

```bash
python3 scripts/evaluate_awgn_sweep.py --steps 600000 --repeats 10
```

Plot the result:

```bash
python3 scripts/plot_fig4.py
```

## Tests

```bash
python3 -m unittest discover -s tests -v
```

## Notes

This implementation focuses on the CIFAR-10 DeepJSCC baseline. It does not include JPEG/JPEG2000, LDPC, or modulation baselines from the original paper.
