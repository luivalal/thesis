#################################################################################################
############ USING MOSTLY CHUNKS OF CODE EXTRACTED FROM BRAINDECODE WEBSITE ######################
##################################################################################################
# Loading and preprocessing the dataset
# -------------------------------------
#
# Loading and preprocessing stays the same as in the `Trialwise decoding
# tutorial <./plot_bcic_iv_2a_moabb_trial.html>`__.

from braindecode.datasets import MOABBDataset
from braindecode.datasets import BNCI2014001
from braindecode.datasets.moabb import KEIO_BCI

subject_id = 3

dataset = KEIO_BCI(7)

####################################################################################
#
#
#

from braindecode.preprocessing import (
    exponential_moving_standardize, preprocess, Preprocessor)
from numpy import multiply

low_cut_hz = 4.  # low cut frequency for filtering
high_cut_hz = 38.  # high cut frequency for filtering
# Parameters for exponential moving standardization
factor_new = 1e-3
init_block_size = 1000
# Factor to convert from V to uV
factor = 1e6

preprocessors = [
    Preprocessor('pick_types', eeg=True, meg=False, stim=False),  # Keep EEG sensors
    Preprocessor(lambda data: multiply(data, factor)),  # Convert from V to uV
    Preprocessor('filter', l_freq=low_cut_hz, h_freq=high_cut_hz),  # Bandpass filter
    Preprocessor(exponential_moving_standardize,  # Exponential moving standardization
                 factor_new=factor_new, init_block_size=init_block_size)
]

# Transform the data
preprocess(dataset, preprocessors)


# We first choose the compute/input window size that will be fed to the
# network during training This has to be larger than the networks
# receptive field size and can otherwise be chosen for computational
# efficiency (see explanations in the beginning of this tutorial). Here we
# choose 1000 samples, which are 4 seconds for the 250 Hz sampling rate.
#

input_window_samples = 2000

import torch
from braindecode.util import set_random_seeds
from braindecode.models import ShallowFBCSPNet


cuda = torch.cuda.is_available()  # check if GPU is available, if True chooses to use it
device = 'cuda' if cuda else 'cpu'
if cuda:
    #torch.backends.cudnn.benchmark = True
    pass
# Set random seed to be able to roughly reproduce results
# Note that with cudnn benchmark set to True, GPU indeterminism
# may still make results substantially different between runs.
# To obtain more consistent results at the cost of increased computation time,
# you can set `cudnn_benchmark=False` in `set_random_seeds`
# or remove `torch.backends.cudnn.benchmark = True`
seed = 20200220
set_random_seeds(seed=seed, cuda=cuda)

n_classes = 4
# Extract number of chans from dataset
n_chans = dataset[0][0].shape[0]

model = ShallowFBCSPNet(
    n_chans,
    n_classes,
    input_window_samples=input_window_samples,
    final_conv_length=30,
)

# Send model to GPU
if cuda:
    model.cuda()


#####################################################################
from braindecode.models import to_dense_prediction_model, get_output_shape

to_dense_prediction_model(model)
n_preds_per_input = get_output_shape(model, n_chans, input_window_samples)[2]


####################################################################################
####################################################################################
from braindecode.preprocessing import create_windows_from_events

trial_start_offset_seconds = -0.5

###############################################################
# Extract sampling frequency, check that they are same in all datasets
sfreq = dataset.datasets[0].raw.info['sfreq']

##################################################################
assert all([ds.raw.info['sfreq'] == sfreq for ds in dataset.datasets])

# Calculate the trial start offset in samples.
trial_start_offset_samples = int(trial_start_offset_seconds * sfreq)

# Create windows using braindecode function for this. It needs parameters to define how
# trials should be used.
windows_dataset = create_windows_from_events(
    dataset,
    trial_start_offset_samples=trial_start_offset_samples,
    trial_stop_offset_samples=0,
    window_size_samples=input_window_samples,
    window_stride_samples=n_preds_per_input,
    drop_last_window=False,
    preload=True
)

splitted = windows_dataset.split('session')
train_set = splitted['session_T']
valid_set = splitted['session_E']

#########################################################################################
#########################################################################################

# .. note::
#
#     In this tutorial, we use some default parameters that we
#     have found to work well for motor decoding, however we strongly
#     encourage you to perform your own hyperparameter optimization using
#     cross validation on your training data.
#

from skorch.callbacks import LRScheduler
from skorch.helper import predefined_split

from braindecode import EEGClassifier
from braindecode.training import CroppedLoss

# These values we found good for shallow network:
#lr = 0.0625 * 0.01
# lr = 1 * 0.01
# weight_decay = 0

# For deep4 they should be:
lr = 1.35 * 0.01
weight_decay = 0.5 * 0.001

batch_size = 64
n_epochs = 40           

###################                 TO INCREASE PERFORMANCE: CHANGE LR  &  EPOCHS                         ##################################################

clf = EEGClassifier(
    model,
    cropped=True,
    criterion=CroppedLoss,
    criterion__loss_function=torch.nn.functional.nll_loss,
    optimizer=torch.optim.AdamW,
    train_split=predefined_split(valid_set),
    optimizer__lr=lr,
    optimizer__weight_decay=weight_decay,
    iterator_train__shuffle=True,
    batch_size=batch_size,
    callbacks=[
        "accuracy", ("lr_scheduler", LRScheduler('CosineAnnealingLR', T_max=n_epochs - 1)),
    ],
    device=device,
)
# Model training for a specified number of epochs. `y` is None as it is already supplied
# in the dataset.
clf.fit(train_set, y=None, epochs=n_epochs)



###################################################################################################
###################################################################################################

y_true = valid_set.get_metadata().target
y_pred = clf.predict(valid_set)

###################################################################################################
###################################################################################################

my_file = open('My_results00.txt', 'w')

for item in y_pred:
    my_file.write(str(item) + '\n')

my_file.write('4' + '\n')

my_file.close()

print('length of the eval set is')
print(len(valid_set))









from sklearn.metrics import confusion_matrix
confusion_mat = confusion_matrix(y_true, y_pred)

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8, 6))
sns.heatmap(confusion_mat, annot=True, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()





###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################
##################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################



# #!/usr/bin/env python

# import rospy
# from std_msgs.msg import Int32

# def publish_numbers():
#     rospy.init_node('number_publisher', anonymous=True)
#     pub = rospy.Publisher('number_topic', Int32, queue_size=10)
#     rate = rospy.Rate(1)  # Adjust the publishing rate as per your requirement

#     file_path = 'My_results00.txt'  # Replace with the actual file path

#     try:
#         with open(file_path, 'r') as file:
#             for line in file:
#                 number = int(line.strip())
#                 pub.publish(number)
#                 rate.sleep()
#     except IOError:
#         rospy.logerr('Failed to open the file.')

# if __name__ == '__main__':
#     try:
#         publish_numbers()
#     except rospy.ROSInterruptException:
#         pass

