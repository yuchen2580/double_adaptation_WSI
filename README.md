# Double adversarial domain adaptation for whole-slide-imageclassification

This is the code for our MIDL submission.

This repo requies CUDA to run.
The libararies are listed in requirements.txt. 
You will need to use conda to create a virtual env with this file.
```sh
conda create --name <env name> --file requirements.txt
```

Another requirement is to download the [Pydmed](https://github.com/amirakbarnejad/PyDmed) and place it to the  same folder of this repo. 

The train steps are in 'train.ipynb' under 'DA_private_warwick' folder.

The test steps are in 'collect_stats_warwickher2.ipynb' under 'DA_private_warwick' folder.
