基于ResNet50迁移学习的垃圾分类

---------
--label
--npy
--ResNet50迁移
--train_data
--UI界面
---------
数据集来自华为垃圾数据集，百度可搜到
链接：https://www.heywhale.com/mw/dataset/5f8eadade48a3f00302f7299/file

label存放数据集标签  train_data存放数据集图片
npy文件夹需要新建，运行make_npy.py文件，用于存放编码后的文件

运行make_label.py和make_npy.py文件
生成对应文件后，运行训练文件

本项目使用内置库和自行搭建两种版本
内置库要训练的快一些
