Instruction for running the code:

1. Download all the directories from the box link.
2. Run the MATLAB file "main.m". It will create training variables to train the network.
	=> Specify in the "noise_type" variable for which type of noise, speech or non-speech type, you want to create training variables
3. Run the code "python_training.py". It will train the network for specified network parameters mentioned inside the code.
4. After training is done, run "test_file.m". Specify for which type of noise, you are doing testing. It will show
	a) Latency of the testing
	b) Time domain plot of the testing and denoised speech
	c) Frequency diagram of the testing and denoised speech
	