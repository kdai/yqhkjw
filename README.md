<<<<<<< HEAD
##### 数据处理
-   step1: 音频-视频分离，将音频从新的视频中分离出来。分离的代码在visual-audio fusion/tool.py中的youtube_splitAudio()方法，在该文件中选择该方法运行，参数1是视频的地址，参数2是分离出来的音频要存放的位置
运行脚本：
```python
cd visual-audio fusion
python tool.py
```
该方法已经集成了如下功能：不重复提取已经存在的音频、自动分类放好不同domain的音频。也就是新下载的视频只要放到相同domain里的文件夹就好。

-   step2: 提取音频特征
运行脚本：(如提取skating)
```py
cd audio_process
python AudioCNN.py --ctg skating --dataset instagram

parameter: ctg，是种类的意思，相当于domain
parameter: dataset，数据集，在这里，训练集是instagram, 测试集有youtube、tvsum、cosum。
还是不要用显卡的比较好，因为不知道音频的长度，很容易爆显存，所以用内存提就可以了，但同一台机不宜跑太多这种程序
```

-   step3: 提取视频频特征
运行脚本：(如提取skating)
```py
cd video-classifcation-3d-cnn-pytorch
CUDA_VISIBLE_DEVICES=1 python main.py --video_root /home/share/Highlight/orgDataset/instagram/skating --savename /home/share/Highlight/proDataset/TrainingSet/skating_1s.npy --model ./resnet-34-kinetics.pth --mode feature 

主要参数：（提取时需要根据domain进行修改）
parameter: video_root, 对应domain的视频位置
parameter: savename，保存特征文件的位置
```

-   step4: 提取视频长度
运行脚本：(如提取skating)
```py
cd audio_process
python getDuration --ctg skating --dataset instagram

参数同步骤2，不需要用到显卡
```


##### 显卡异常处理
当使用ctrl+C停止程序是，可能会出程序已经停止，但显卡是仍然显示自己的程序在占用，这是遗留下来的defunct的程序，使用以下脚本可以清除：
```py
比如第0张显卡
step 1: 查出进程id号
fuser -v /dev/nvidia0
(如果是第1张显卡，则为：fuser -v /dev/nvidia1)

step 2: 杀死进程
kill -9 id
```

##### 主程序运行：
```py
cd visual-audio fusion
CUDA_VISIBLE_DEVICES=1 python trainMIL.py --dataset youtube --domain gymnastics --train_path /home/share/Highlight/proDataset/TrainingSet/ --test_path /home/share//Highlight/proDataset/DomainSpecific --topk_mAP 1 --FNet MILModel3 --HNet HNet1 --AM AttentionModule_1 --DS MILDataset --AHLoss AdaptiveHingerLoss --short_lower 10 --short_upper 40 --long_lower 60 --long_upper 60000 --bagsize 60 

parameter: dataset, 测试数据集
parameter: domain, 训练/测试 domain
parameter: train_path, 训练集的位置
parameter: test_path, 测试集的位置
parameter: topk_mAP, top K的mAP,youtube为top1 ,tvsum为top5
parameter: FNet, score估计模型
parameter: HNet, 以前的估计w值，现在无用参数，可去掉
parameter: AM, vision和audio的fusion模型
parameter: DS, dataset模型
parameter: AHLoss, 使用的hingerloss

```

##### 注意事项
上面所有的脚本都可以在对应文件夹里的.sh文件找到，在做实验的过程中，要记录好使用脚本的的命令，不允许在代码opts.py中修改默认参数，所有参数的修改都应该通过代码运行时的命令修改（如同上面运行脚本的那些变量），做实验要记录，哪些脚本的实验结果是如何的都应该记录下来，可以用一个excel表来记录。

注意：所有的代码不可修改，只允许新增。比如有了新的想法，产生新的网络结构，可以在model.py里面重新写一个网络，不允许在原先网络上修改，然后可以通过--FNet参数来替换新的网络。当trainXXX.py函数实在需要很大变动时，应该重新新建一个文件，并在文件的顶部声明做了哪些修改。

以上提到的东西主要是为了保证实验的可溯性。

=======
# yqhkjw
>>>>>>> 163df7f0d21dfed528c8654d0eeb12992f0dcffa
