classify_text_with_bert.ipynb is the ORIGINAL file retried form the TF website.
electra_modified.ipynb is a modification of this file, and utilizes a different sequential architecture,
A more detailed explanation of the architecture choices are explained in the write-up.

In order to use the following files, you should download the required Data and Models from here:
https://drive.google.com/drive/folders/1HDwiN1V25_uOkTUApsER5mJCQ7ivOOFe?usp=sharing

Additionally, training the following models may take a significant amount of time:
    Laptop/Macbook -> ~1 hour per epoch
    Desktop without cuDNN -> ~30 minutes per epoch
    Desktop with cuDNN, dGPU -> ~3 minutes per epoch

The implementation of the BERT-based models must overcome several hurdles before even beginning the implementation – namely, access computational power and the proper dependencies. While most machines can run Python code, specialized dependencies are required to run the code at any significant speed. This can be avoided by running code on Google Collab, but the resources provided are equivalent to a single GPU, and frequently result in issues when trying to debug or refactor code. Enabling matrix multiplication on a GPU or any massively parallel operations requires CUDA and cuDNN for Nvidia cards and ROCm for AMD based cards. Installation is not difficult, but can be extraordinarily time-consuming.

Even with access to accelerated computation, we are unable to replicate results from academic studies of neural networks; retraining BERT takes on the order of days to weeks even with dedicated server hardware, so replicating BERT’s results falls out of the scope of this project. Even with non-small BERT models, the VRAM requirement exceeds the 10GB limit found on most consumer graphics cards as well as Google's TPU allocations for non-authenticated users.