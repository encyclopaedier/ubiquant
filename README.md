# ubiquant

**各个模块作用**

Run：入口函数，直接运行即可

arg_parser:用于控制台传参，方便在aws云服务器上跑的，后面再改（未完成）

Dataloader：dataloader，按照hw1p2的方式写的，每次从小批次的csv里面读取数据，好处是电脑内存小也可以读，方便debug（已完成），会考虑多线程dataloader（未完成）

config：config，设置一些训练/debug的configuration，比如数据集路径，epoch数量，debug路径，batch_size数量之类的

Model：各种model，同时用于存放跑好的model

preprocess：用于将train数据集split成小的数据集，给dataloader读（已完成），分割train dataset&validation dataset（未完成）

train_eval:训练的主函数，train为训练主要函数，init_train_env配置训练环境（比如optimizer之类的）

**流程**

arg_parser从控制台读参数（暂时没用到），调用入口函数（main），main调用train_eval记进行训练，过程中会调用dataloder，model等代码

**数据集**

数据集地址如下，https://drive.google.com/drive/folders/110U_N-orkPE0g8s4XbfyXM8f87fVrYVr?usp=sharing
其中split_data文件夹是split过，处理好的training文件。暂时没有分validation集.

放到config.config里的TRAINING_PATH即可。
