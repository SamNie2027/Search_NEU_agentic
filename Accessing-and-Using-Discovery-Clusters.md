# Accessing and Using Discovery Clusters at Northeastern University

**Discovery** is a high-performance computing (HPC) resource for the Northeastern University research community. If you need computation resources for your course project, you can apply for access to the Discovery cluster.

## 1. Requesting an Account

To access Discovery, you first need to create an account.

You can request an account through **[ServiceNow](https://service.northeastern.edu/tech?id=sc_cat_item&sys_id=0ae24596db535fc075892f17d496199c)**. You will need your **Northeastern username and password** to log in.

### Steps to Request an Account

1. Go to the **ServiceNow Research Computing Access Request Form**.
2. Enter your details.
3. Under **Affiliation with Northeastern University**, select **Undergrad Student**.
4. Under **University Sponsor**, type or select your professor’s name.
5. In the **Gaussian** field, select **Yes**.
6. Submit the form.

You should receive a confirmation email from **ITS** within 24 hours.

## 2. Connecting to Discovery via SSH

Once your account is approved, you can connect to Discovery using an SSH session.

### For macOS Users

You can use **Terminal** to connect to Discovery via SSH.

If you need to use software with a graphical interface (GUI), such as **Matlab** or **Maestro**, include the `-Y` option.

**Command:**

```
ssh <username>@login.discovery.neu.edu
```

Replace `<username>` with your **Northeastern username**.

Then, type your Northeastern password and press **Enter**.

> 💡 See the [official instructions](https://rc-docs.northeastern.edu/en/1.2.0/first_steps/connect_mac.html) for setting up SSH for macOS 11 or configuring passwordless SSH.

### For Windows Users

Before connecting to Discovery, you’ll need a terminal program such as **MobaXterm** or **PuTTY**.
 We recommend **MobaXterm** because it supports both SSH and file transfer.

#### Connecting with MobaXterm

1. Open **MobaXterm**.

2. Click **Session** → select **SSH** as the connection type.

3. In **Remote Host**, enter:

   ```
   login.discovery.neu.edu
   ```

   Make sure the **Port** is set to `22`.

4. Click **OK**.

5. At the prompt, type your **Northeastern username** and press **Enter**.

6. Type your **password** and press **Enter**.

> Note: The cursor will not move as you type your password — this is expected behavior. See [the instruction in this link](https://rc-docs.northeastern.edu/en/1.2.0/first_steps/connect_windows.html).

## 3. Installing a Conda Environment

It’s recommended to install a **Conda environment** for package and environment management.

### Installing Miniconda

1. If you’re on a login node, move to a compute node:

   ```
   srun --partition=short --nodes=1 --cpus-per-task=1 --pty /bin/bash
   ```

2. Download the latest Miniconda installer:

   ```
   wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
   ```

3. Verify the installer’s hash:

   ```
   sha256sum Miniconda3-latest-Linux-x86_64.sh
   ```

4. Install Miniconda (replace `<dir>` with your installation directory):

   ```
   bash Miniconda3-latest-Linux-x86_64.sh -b -p <dir>
   ```

   Example (recommended):

   ```
   /work/<mygroup>/<mydirectory>/miniconda3
   ```

5. Activate the Conda environment:

   ```
   source <dir>/bin/activate
   ```

6. Update Conda (if you own the Conda installation):

   ```
   conda update conda -y
   ```

> 🔗 Follow [this link](https://rc-docs.northeastern.edu/en/1.2.0/first_steps/bashrc.html) for instructions on setting up your shell environment for Conda, so that `conda` commands are available directly after login.