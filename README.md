# numPredict
predict fire number

* 如何运行程序
安装CNTK
https://docs.microsoft.com/en-us/cognitive-toolkit/setup-cntk-on-your-machine
1. 创建一个目录data放到和numPredict同一个目录下。把FireNJ_address.csv, raw.csv, noaa_climate.csv拷到numPredict目录下。
2. 在命令行中到numPredict目录下运行python dataPrep.py。结果将在numPredict目录下产生data_test.csv和data_train.csv。
3. 运行python train.py, 得到结果
epoch: 0, loss: 10.31042
epoch: 200, loss: 0.96593
epoch: 400, loss: 0.28879
epoch: 600, loss: 0.45312
epoch: 800, loss: 0.19490
epoch: 1000, loss: 0.16596
epoch: 1200, loss: 0.19331
epoch: 1400, loss: 0.14459
epoch: 1600, loss: 0.13115
epoch: 1800, loss: 0.14896
training took 10879.2 sec
train mse: 0.000708
test mse: 0.100145

* 程序说明
通过过去几天（time_steps）的观察数据（温度，降水，火灾数）和下一天气象预测的数据（温度，降水）来预测下一天的火灾数。目前我只用到了温度和降水两个辅助参数。其他如气象类型，湿度，风力都没考虑，原因一是我从noaa下载的数据没有包括这些数据，第二南京全年湿度一般都比较大，所以没有加，风力可能对起火的程度影响更大，但可能不是起火的原因。这个项目只是一个样例，如果有更多的其他辅助数据可以很容易加进去。
几个重要的参数：
EPOCHS - 训练次数，测试程序时设为10减少运行时间。
BATCH_SIZE - 一次训练的数据大小。
time_steps - 使用过去time_step天的数据预测下一天的火灾。

* 参考
1. Multi-step Time Series Forecasting with Long Short-Term Memory Networks in Python
https://machinelearningmastery.com/multi-step-time-series-forecasting-long-short-term-memory-networks-python/

2. CNTK 106: Part A - Time series prediction with LSTM (Basics)
https://www.cntk.ai/pythondocs/CNTK_106A_LSTM_Timeseries_with_Simulated_Data.html