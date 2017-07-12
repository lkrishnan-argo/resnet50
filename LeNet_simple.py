import numpy as np
import os
import shutil
from caffe2.python import core, cnn, net_drawer, workspace
import aetros.backend
job = aetros.backend.start_job('lkrishnan-argo/LeNet')
accuracy_channel = job.create_channel('accuracy', kpi=True, main=True, yaxis={'dtick': 10})

# If you would like to see some really detailed initializations,
# you can change --caffe2_log_level=0 to --caffe2_log_level=-1
core.GlobalInit(['caffe2', '--caffe2_log_level=0'])
# set this where the root of caffe2 is installed
caffe2_root = "~/caffe2"
print("Necessities imported!")

data_folder = os.path.expanduser('~/caffe2_notebooks/tutorial_data/mnist')
def AddInput(model, batch_size, db, db_type):
    # load the data
    data_uint8, label = model.TensorProtosDBInput(
        [], ["data_uint8", "label"], batch_size=batch_size,
        db=db, db_type=db_type)
    # cast the data to float
    data = model.Cast(data_uint8, "data", to=core.DataType.FLOAT)
    # scale data from [0,255] down to [0,1]
    data = model.Scale(data, data, scale=float(1./256))
    # don't need the gradient for the backward pass
    data = model.StopGradient(data, data)
    return data, label
print("Input function created.")

def AddLeNetModel(model, data):
    conv1 = model.Conv(data, 'conv1', 1, 20, 5)
    pool1 = model.MaxPool(conv1, 'pool1', kernel=2, stride=2)
    conv2 = model.Conv(pool1, 'conv2', 20, 50, 5)
    pool2 = model.MaxPool(conv2, 'pool2', kernel=2, stride=2)
    fc3 = model.FC(pool2, 'fc3', 50 * 4 * 4, 500)
    fc3 = model.Relu(fc3, fc3)
    pred = model.FC(fc3, 'pred', 500, 10)
    softmax = model.Softmax(pred, 'softmax')
    return softmax
print("Model function created.")

def AddAccuracy(model, softmax, label):
    accuracy = model.Accuracy([softmax, label], "accuracy")
    return accuracy
print("Accuracy function created.")

def AddTrainingOperators(model, softmax, label):
    # something very important happens here
    xent = model.LabelCrossEntropy([softmax, label], 'xent')
    # compute the expected loss
    loss = model.AveragedLoss(xent, "loss")
    # track the accuracy of the model
    AddAccuracy(model, softmax, label)
    # use the average loss we just computed to add gradient operators to the model
    model.AddGradientOperators([loss])
    # do a simple stochastic gradient descent
    ITER = model.Iter("iter")
    # set the learning rate schedule
    LR = model.LearningRate(
        ITER, "LR", base_lr=-float(job.get_parameter('base_learning_rate')), policy="step", stepsize=1, gamma=0.999 )
    # ONE is a constant value that is used in the gradient update. We only need
    # to create it once, so it is explicitly placed in param_init_net.
    ONE = model.param_init_net.ConstantFill([], "ONE", shape=[1], value=1.0)
    # Now, for each parameter, we do the gradient updates.
    for param in model.params:
        # Note how we get the gradient of each parameter - CNNModelHelper keeps
        # track of that.
        param_grad = model.param_to_grad[param]
        # The update is a simple weighted sum: param = param + param_grad * LR
        model.WeightedSum([param, ONE, param_grad, LR], param)
    # # let's checkpoint every 20 iterations, which should probably be fine.
    # # you may need to delete tutorial_files/tutorial-mnist to re-run the tutorial
    # model.Checkpoint([ITER] + model.params, [],
    #                db="mnist_lenet_checkpoint_%05d.leveldb",
    #                db_type="leveldb", every=20)
print("Training function created.")

def AddBookkeepingOperators(model):
    # Print basically prints out the content of the blob. to_file=1 routes the
    # printed output to a file. The file is going to be stored under
    #     root_folder/[blob name]
    model.Print('accuracy', [], to_file=1)
    model.Print('loss', [], to_file=1)
    # Summarizes the parameters. Different from Print, Summarize gives some
    # statistics of the parameter, such as mean, std, min and max.
    for param in model.params:
        model.Summarize(param, [], to_file=1)
        model.Summarize(model.param_to_grad[param], [], to_file=1)
    # Now, if we really want to be verbose, we can summarize EVERY blob
    # that the model produces; it is probably not a good idea, because that
    # is going to take time - summarization do not come for free. For this
    # demo, we will only show how to summarize the parameters and their
    # gradients.
print("Bookkeeping function created")

train_model = cnn.CNNModelHelper(order="NCHW", name="mnist_train")
data, label = AddInput(
    train_model, batch_size=64,
    db=os.path.join(data_folder, 'mnist-train-nchw-lmdb'),
    db_type='lmdb')
softmax = AddLeNetModel(train_model, data)
AddTrainingOperators(train_model, softmax, label)
# AddBookkeepingOperators(train_model)

# Testing model. We will set the batch size to 100, so that the testing
# pass is 100 iterations (10,000 images in total).
# For the testing model, we need the data input part, the main LeNetModel
# part, and an accuracy part. Note that init_params is set False because
# we will be using the parameters obtained from the train model.
test_model = cnn.CNNModelHelper(
    order="NCHW", name="mnist_test", init_params=False)
data, label = AddInput(
    test_model, batch_size=100,
    db=os.path.join(data_folder, 'mnist-test-nchw-lmdb'),
    db_type='lmdb')
softmax = AddLeNetModel(test_model, data)
AddAccuracy(test_model, softmax, label)

# Deployment model. We simply need the main LeNetModel part.
deploy_model = cnn.CNNModelHelper(
    order="NCHW", name="mnist_deploy", init_params=False)
AddLeNetModel(deploy_model, "data")
# You may wonder what happens with the param_init_net part of the deploy_model.
# No, we will not use them, since during deployment time we will not randomly
# initialize the parameters, but load the parameters from the db.

print('Created training and deploy models.')



# The parameter initialization network only needs to be run once.
workspace.RunNetOnce(train_model.param_init_net)
# creating the network
workspace.CreateNet(train_model.net)
# set the number of iterations and track the accuracy & loss
total_iters = 200
accuracy = np.zeros(total_iters)
loss = np.zeros(total_iters)
# Now, we will manually run the network for 200 iterations. 
for i in range(total_iters):
    workspace.RunNet(train_model.net.Proto().name)
    accuracy[i] = workspace.FetchBlob('accuracy')
    loss[i] = workspace.FetchBlob('loss')
    if i%20 == 0:
    	print('iter {}: Loss={:.2}'.format(i, loss[i]))
    	accuracy_channel.send(i, accuracy[i])
    job.progress(i, total=total_iters)
# run a test pass on the test net
workspace.RunNetOnce(test_model.param_init_net)
workspace.CreateNet(test_model.net)
test_accuracy = np.zeros(100)
for i in range(100):
    workspace.RunNet(test_model.net.Proto().name)
    test_accuracy[i] = workspace.FetchBlob('accuracy')
# After the execution is done, let's plot the values.
print('test_accuracy: %f' % test_accuracy.mean())

job.done()