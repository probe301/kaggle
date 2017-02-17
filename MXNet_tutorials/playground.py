import mxnet as mx


batch_size = 100

# Create a place holder variable for the input data
data = mx.sym.Variable('data')
# Flatten the data from 4-D shape (batch_size, num_channel, width, height)
# into 2-D (batch_size, num_channel*width*height)
data = mx.sym.Flatten(data=data)

# The first fully-connected layer
fc1  = mx.sym.FullyConnected(data=data, name='fc1', num_hidden=128)
# Apply relu to the output of the first fully-connnected layer
act1 = mx.sym.Activation(data=fc1, name='relu1', act_type="relu")

# The second fully-connected layer and the according activation function
fc2  = mx.sym.FullyConnected(data=act1, name='fc2', num_hidden=64)
act2 = mx.sym.Activation(data=fc2, name='relu2', act_type="relu")

# The thrid fully-connected layer,
# note that the hidden size should be 10, which is the number of unique digits
fc3  = mx.sym.FullyConnected(data=act2, name='fc3', num_hidden=10)
# The softmax and loss layer
mlp  = mx.sym.SoftmaxOutput(data=fc3, name='softmax')

# We visualize the network structure with output size (the batch_size is ignored.)
shape = {"data": (batch_size, 1, 28, 28)}
plot = mx.viz.plot_network(symbol=mlp, shape=shape)
plot.view()


# %MXNET_HOME%\lib;%MXNET_HOME%\3rdparty\cudnn\bin;
# %MXNET_HOME%\3rdparty\cudart;
# %MXNET_HOME%\3rdparty\vc;
# %MXNET_HOME%\3rdparty\gnuwin;
# %MXNET_HOME%\3rdparty\openblas\bin;
# C:\Program Files (x86)\Intel\iCLS Client\;
# C:\Program Files\Intel\iCLS Client\;
# C:\Windows\system32;
# C:\Windows;
# C:\AppStartup\;
# C:\Windows\System32\Wbem;
# C:\Windows\System32\WindowsPowerShell\v1.0\;
# C:\Program Files (x86)\NVIDIA Corporation\PhysX\Common;
# C:\Program Files (x86)\Intel\Intel(R) Management Engine Components\DAL;
# C:\Program Files\Intel\Intel(R) Management Engine Components\DAL;
# C:\Program Files (x86)\Intel\Intel(R) Management Engine Components\IPT;
# C:\Program Files\Intel\Intel(R) Management Engine Components\IPT;
# C:\Program Files\Intel\WiFi\bin\;
# C:\Program Files\Common Files\Intel\WirelessCommon\;
# C:\ProgramData\Anaconda3;
# C:\ProgramData\Anaconda3\Scripts;
# C:\ProgramData\Anaconda3\Library\bin;
# C:\Program Files\Git\cmd;
# C:\Program Files\nodejs\;
# C:\Program Files (x86)\Windows Kits\10\Windows Performance Toolkit\;
# C:\Users\plane\AppData\Local\Microsoft\

