'''Process time series and train AI models.'''
import dill
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable
import time
import pandas as pd
import numpy as np
import torch.nn as nn
from Models.LSTM import LSTM
#import mathplotlib as

def train_models(args,Datafile,model_name):

	print("Training model...")
	
	#load data
	with open(Datafile, 'rb') as f:
		dataset = dill.load(f)
	trainX,trainY,testX,testY= dataset[0],dataset[1],dataset[2],dataset[3]
	train_dataset = TensorDataset(torch.Tensor(trainX), torch.Tensor(trainY))
	test_dataset = TensorDataset(torch.Tensor(testX), torch.Tensor(testY))
	train_loader = DataLoader(dataset=train_dataset ,batch_size=300,shuffle=False,num_workers=0)
	test_loader = DataLoader(dataset=test_dataset,batch_size=300,shuffle=False,num_workers=0)

	#load model
	if args.model == "LSTM":
		# model = TS_LSTM(trainX,trainY,testX,testY)
		# model.save(model_name)
				   
		input_dim = len(trainX[0, 0, :])
		output_dim = 1
		hidden_dim = 20  
		torch.set_default_tensor_type(torch.FloatTensor)
		device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
		
		epochs = 200  
		learning_rate = 0.003 
		
		start = time.perf_counter()  

		multi_times = 1  
		for times in range(multi_times):        
			model = LSTM(input_dim, hidden_dim, output_dim).to(device)
			if times == 0:
				print(model)  
			optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
			criterion = nn.MSELoss() 

			for epoch in range(epochs):
				for i, (train_sequence, train_forecast) in enumerate(train_loader): 
					train_sequence = Variable(train_sequence).to(device)
					train_forecast = Variable(train_forecast).to(device)

					# Forward pass
					optimizer.zero_grad() #zero gradients
					outputs = model(train_sequence) #input sequence into model
					loss = criterion(outputs,train_forecast) # calculate MSE 
					#experiment.log_metric("loss",loss,step = epoch) #log the results 

					# Backward and optimize
					loss.backward()
					optimizer.step()
					optimizer.zero_grad()

				if (epoch + 1) % 20 == 0:
					print(f'epoch {epoch + 1}, loss {loss}')

		end = time.perf_counter()  # 运行结束时间
		runTime = end - start
		print("Run time: ", runTime)  # 输出运行时间
		
		torch.save(model,model_name)
		
	'''
	#test       
	prediction=model.predict(testX)

	#data post-processing
	prediction_copies_array = np.repeat(prediction,df.shape[1], axis=-1)
	pred = scaler.inverse_transform(np.reshape(prediction_copies_array,(len(prediction),df.shape[1])))[:,0]
	original_copies_array = np.repeat(testY,5, axis=-1)
	original = scaler.inverse_transform(np.reshape(original_copies_array,(len(testY),5)))[:,0]

	#cal accuracy
	print("Pred Values-- " ,pred,'\n')
	print("Original Values-- ",original,'\n')
	rmse=sqrt(mean_squared_error(pred, original))
	print("rmse:", rmse)

	res_df = pd.DataFrame(list(zip(pred,original)),columns=['prediction','original'])
	res_df=res_df.astype(str)
	res_name="./Results/"+args.DataName+"_"+args.model+str(args.step)+"_"+str(rmse)+".csv"
	res_df.to_csv(res_name,index=False)
	

	plt.plot(original, color = 'red', label = 'Real  Stock Price')
	plt.plot(pred, color = 'blue', label = 'Predicted  Stock Price')
	plt.title(' Stock Price Prediction')
	plt.xlabel('Time')
	plt.ylabel(' Stock Price')
	plt.legend()
	plt.show()
	'''
