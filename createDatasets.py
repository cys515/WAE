'''Create multivariable time series'''
import os
import numpy as np
import pandas as pd
import dill
from sklearn.preprocessing import MinMaxScaler
import timesynth as ts
import matplotlib.pyplot as plt

def plotData(Datafile,args):
	df = pd.read_csv(Datafile)
	df = df[-300:]
	fig, ax = plt.subplots(args.features,1,figsize=(15, 15),constrained_layout=True)

	for i in range(args.features):
		yname = 'var'+str(i)
		df.plot(x='Date', y=yname,marker='o', markersize=1,ax=fig.axes[i])
		ax[i].legend(loc='upper right',fontsize = 15)
		ax[i].set_xlabel("Time", fontsize = 15)
		ax[i].set_ylabel("Value", fontsize = 15)

	plt.subplots_adjust(hspace=0.23)
	fig.suptitle('Dataset '+args.DataName, fontsize=15, y= 0.91)
	fig.savefig("./Fig/"+args.DataName+".png",bbox_inches='tight',pad_inches=0.1)
	plt.close()

def createData(args,Datafile):
	print("Creating Dataset...")

	if args.correlation == 0:
		#all features are independent
		df = pd.DataFrame()
		time_sampler = ts.TimeSampler()
		time = time_sampler.sample_regular_time(num_points=args.times)
		df["Date"] = time
		if args.DataName == 'L1':
			for i in range(args.features):	
				sample = np.sin(time*30)
				df['var'+str(i)] = sample
		if args.DataName == 'L2':
			for i in range(args.features):	
				sample = np.sin(time*30)
				df['var'+str(i)] = sample
		#G1:target~N(0,1),var1~N(0,1),var2~N(0,1)
		if args.DataName == 'G1':
			for i in range(args.features):	
				sample = np.random.normal(0,1,args.times)
				df['var'+str(i)] = sample
		#G2:target~N(0,1),var1~N(0,10),var2~N(0,0.5)
		if args.DataName == 'G2':
			var = [1,10,0.5]
			for i in range(args.features):
				sample = np.random.normal(0,var[i%args.features],args.times)
				df['var'+str(i)] = sample
		#G1:target~N(0,1),var1~N(0,1),var2~N(0,1)
		if args.DataName == 'G3':
			for i in range(args.features):	
				sample = np.random.normal(0,1,args.times)
				np.random.shuffle(sample)
				df['var'+str(i)] = sample
		#P1:target~P(1,1,0.01,0.5),var1~P(1,1,0.01,0.5),var2~P(1,1,0.01,0.5)
		if args.DataName == 'P1':			
			for i in range(args.features):
				pseudo_periodic = ts.signals.PseudoPeriodic(freqSD = 0.01,ampSD=0.5) 
				timeseries_pp = ts.TimeSeries(pseudo_periodic)
				sample, signals_pp, errors_pp = timeseries_pp.sample(time)
				df['var'+str(i)] = sample		
		#P2:target~P(1,1,0.01,0.5),var1~(0.5,1,0.01,0.5),var2~P(2,1,0.01,0.5)
		if args.DataName == 'P2':
			var = [100,50,200]
			for i in range(args.features):
				pseudo_periodic = ts.signals.PseudoPeriodic(frequency = var[i%args.features],freqSD = 0.01,ampSD=0.5) 
				timeseries_pp = ts.TimeSeries(pseudo_periodic)
				sample, signals_pp, errors_pp = timeseries_pp.sample(time)
				df['var'+str(i)] = sample	
		#P3:target~P(1,1,0.01,0.5),var1~(1,1,0.1,0.5),var2~P(1,1,0.01,0.01)	
		if args.DataName == 'P3':
			for i in range(args.features):
				if i%args.features == 0:
					pseudo_periodic = ts.signals.PseudoPeriodic(freqSD = 0.01,ampSD=0.5) 
					timeseries_pp = ts.TimeSeries(pseudo_periodic)
					sample, signals_pp, errors_pp = timeseries_pp.sample(time)
					df['var'+str(i)] = sample
				if i%args.features == 1:
					pseudo_periodic = ts.signals.PseudoPeriodic(freqSD = 0.1,ampSD=0.5) 
					timeseries_pp = ts.TimeSeries(pseudo_periodic)
					sample, signals_pp, errors_pp = timeseries_pp.sample(time)
					df['var'+str(i)] = sample
				if i%args.features == 2:
					pseudo_periodic = ts.signals.PseudoPeriodic(freqSD = 0.01,ampSD=1) 
					timeseries_pp = ts.TimeSeries(pseudo_periodic)
					sample, signals_pp, errors_pp = timeseries_pp.sample(time)
					df['var'+str(i)] = sample
		#P4:target~P(1,1,0.01,0.5),var1~P(1,1,0.01,0.5),var2~P(1,1,0.01,0.5)
		if args.DataName == 'P4':			
			for i in range(args.features):
				red_noise = ts.noise.RedNoise(std=0.5, tau=0.8)
				pseudo_periodic = ts.signals.PseudoPeriodic(freqSD = 0.01,ampSD=0.5) 
				timeseries_pp = ts.TimeSeries(pseudo_periodic, noise_generator=red_noise)
				sample, signals_pp, errors_pp = timeseries_pp.sample(time)
				np.random.shuffle(sample)
				df['var'+str(i)] = sample		
		#P4:target~P(1,1,0.01,0.5),var1~P(1,1,0.01,0.5),var2~P(1,1,0.01,0.5)
		if args.DataName == 'P5':			
			for i in range(args.features):
				irregular_time_samples = time_sampler.sample_irregular_time(num_points=args.times*2, keep_percentage=50)
				noise = ts.noise.GaussianNoise(std=0.3)
				pseudo_periodic = ts.signals.PseudoPeriodic(freqSD = 1,ampSD=0.5) 
				timeseries_pp = ts.TimeSeries(pseudo_periodic, noise_generator=noise)
				sample, signals_pp, errors_pp = timeseries_pp.sample(irregular_time_samples)
				np.random.shuffle(sample)
				df['var'+str(i)] = sample				
		#P3:target~P(1,1,0.01,0.5),var1~(1,1,0.1,0.5),var2~P(1,1,0.01,0.01)	
		if args.DataName == 'C1':
			for i in range(args.features):
				car = ts.signals.CAR(ar_param=0.9, sigma=0.01, start_value=1.0)
				car_series = ts.TimeSeries(signal_generator=car)
				sample, signals_pp, errors_pp = car_series.sample(time)
				df['var'+str(i)] = sample
		if args.DataName == 'C2':
			for i in range(args.features):
				car = ts.signals.CAR(ar_param=0.9, sigma=0.01, start_value=1.0)
				car_series = ts.TimeSeries(signal_generator=car)
				sample, signals_pp, errors_pp = car_series.sample(time)
				np.random.shuffle(sample)
				df['var'+str(i)] = sample				
		
		df.astype(float)
		df.to_csv(Datafile,index=False)
		plotData(Datafile,args)
			
	
def createDill(args,Datafile):

	if not os.path.exists(Datafile+'.csv'):
		createData(args,Datafile+'.csv')

	#data process
	df=pd.read_csv(Datafile+'.csv').drop(columns = "Date")
	
	#devide data,train:test=4:1
	test_split = round(len(df)*0.20)
	df_for_training = df[:-test_split]
	df_for_testing = df[-test_split:]

	#normalization
	scaler = MinMaxScaler(feature_range=(0,1))
	df_for_training_scaled = scaler.fit_transform(df_for_training)
	df_for_testing_scaled = scaler.transform(df_for_testing)

	#data transform to [X_step,Y]
	def createXY(dataset,n_past):
		dataX = []
		dataY = []
		for i in range(n_past, len(dataset)):
			dataX.append(dataset[i - n_past:i, 0:dataset.shape[1]])
			dataY.append(dataset[i,0])
		return np.array(dataX),np.array(dataY)        

	trainX,trainY=createXY(df_for_training_scaled,args.step) #df_X=[len(train)-step,step,feature_num],df_Y=[len(train)-step]
	testX,testY=createXY(df_for_testing_scaled,args.step)

	dataset = [trainX,trainY,testX,testY]

	with open(Datafile+str(args.step)+'.dill', 'wb') as f:
		dill.dump(dataset, f)
