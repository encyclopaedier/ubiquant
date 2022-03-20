# ubiquant



**各个模块作用**

入口函数还没写

arg_parser:用于控制台传参，方便在aws云服务器上跑的，后面再改

Dataloader：dataloader，按照hw1p2的方式写的，每次从小批次的csv里面读取数据，好处是电脑内存小也可以读，方便debug（已完成），会考虑多线程dataloader（未完成）

config：config，设置一些训练/debug的configuration，比如数据集路径，epoch数量，debug集数量之类的

Model：各种model，同时用于存放跑好的model

preprocess：用于将train数据集split成小的数据集，给dataloader读（已完成），分割train dataset&validation dataset（未完成）

train_eval:训练的主函数，参照hw1p2格式写的（未完成）

**流程**

arg_parser从控制台读参数，调用入口函数（main），main调用train_eval记进行训练，过程中会调用dataloder，model等代码

