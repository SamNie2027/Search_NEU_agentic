# Set up a Local Python Environment

If you plan to run the code on your local laptop or workstation, you’ll need to install several required libraries. We recommend using **Python 3.8 or higher** to ensure compatibility with the latest packages. 

### 1. Installing Conda

Conda is a tool for managing Python environments and dependencies. We recommend **Miniconda**, a lightweight version of Conda that includes only the essentials.

1. Visit the official Miniconda download page: https://docs.conda.io/en/latest/miniconda.html

2. Choose the installer that matches your operating system:

   - **Windows:** Download the `.exe` installer for your system (64-bit recommended).
   - **macOS:** Download the `.pkg` installer (Apple Silicon or Intel version).
   - **Linux:** Download the `.sh` installer.

3. Run the installer:

   - **Windows:** Double-click the `.exe` file and follow the prompts.

   - **macOS / Linux:** Open a terminal and run the following command (replace `<installer-name>` with the downloaded file name):

     ```
     bash <installer-name>.sh
     ```

4. Follow the on-screen instructions and restart your terminal after installation.

5. Verify that Conda is installed by running:

   ```
   conda --version
   ```

   You should see output like:

   ```
   conda 24.5.0
   ```

### 2. Create a Python Environment

You can use **Conda** to manage environments and install packages. To create a new environment (you can replace `agent_env` with any name you prefer):

```
conda create --name agent_env python=3.8
conda activate agent_env
```

### 3. Install Required Libraries

This project requires the following Python libraries:

#### **PyTorch**

You can install the latest version of PyTorch based on your system configuration.

- **If you have an NVIDIA GPU with CUDA 12.1:**

  ```
  conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
  ```

- **If you don’t have a supported GPU (CPU-only build):**

  ```
  conda install pytorch torchvision torchaudio cpuonly -c pytorch
  ```

For installation on other platforms, see the official PyTorch installation page.

#### **Hugging Face Packages**

We will also use several libraries from **Hugging Face** for working with pretrained language models:

```
pip install transformers datasets accelerate peft trl
```