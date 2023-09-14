'''Explain time series and AI model with different explaining models.'''
import dill
import numpy as np
import os
import torch
from XAI.SHAP import ShapTS
from XAI.LIME import LimeTS
from XAI.segment import SegmentationPicker


def explain(args,Datafile,model_name):

	device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

	explain_name = './Results/' + args.DataName + '_' + args.model + '_' + str(args.step) + '_explain.dill'
	rel_res = args.explain + '_' + args.segment + '_' + args.segpara

	print("Explain data "+Datafile+" with black box model "+model_name+" with "+args.explain+'-'+args.segment+" algorithm.")
	#load test data      
	with open(Datafile, 'rb') as f:
		dataset = dill.load(f)

	X = dataset[2][0:20]
	y = dataset[3][0:20]
	
	relevances_data = {'samples':X,'labels':y}
	if os.path.exists(explain_name):
		with open(explain_name, 'rb') as f:
			relevances_data = dill.load(f)
		if rel_res in relevances_data.keys():
			return
		
	#load black box model 
	model_name='./Models/'+args.DataName+"_"+args.model+str(args.step)
	# model = keras.models.load_model(model_name)
	model = torch.load(model_name).to(device)

	def predict_fn(x):
		x = torch.from_numpy(x).float().to(device)
		if len(x.shape) == 2:
			# prediction = model.predict(x[np.newaxis]).ravel()
			prediction = model(x[np.newaxis])
		else:
			# prediction = model.predict(x).ravel()
			prediction = model(x,shuffle = False)
		prediction = prediction.cpu().detach().numpy().ravel()
		return prediction
	
	if args.explain[0:1] == 'L':
		relevances = []
		windows = []
		for i, x in enumerate(X):

			explainer = LimeTS(n_samples=200) 

			# set fit model
			explainer._kernel = args.explain

			# estimate partitions and take 10% of time series length
			ts_len = max(x.shape)

			win_length = 20	
			partitions = int(ts_len / win_length)     #n_samples*0.7>partitions*features  

			# set segmentation
			segmentation_class = SegmentationPicker().select(args.segment, partitions, win_length)
			explainer._segmenter = segmentation_class
			
			try:
				explaination, window = explainer.explain(x, predict_fn, segmentation_method=args.segpara) 
			except Exception as e:
				print(e)
				print('Error')
				explaination = None
				window = None 

			relevances.append(explaination)
			windows.append(window)
					
			print('Done:', i)

		seg_name= 'win_' + args.segment + '_' +args.segpara
		relevances_data.update({seg_name:windows,rel_res:relevances})
		
	if args.explain[0:1] == "S":
		relevances = []
		windows = []
		background_data = dataset[0][-20:]
		for i, x in enumerate(X):

			# estimate partitions and take 10% of time series length
			ts_len = max(x.shape)
			partitions = int(ts_len * 0.05)   
			win_length = int(ts_len / partitions)  

			explainer = ShapTS(model = model, background_data = background_data, ts_data = x)
			if args.explain == "SHAP":
				explainer._kernel = "Kernel"

			# set segmentation
			segmentation_class = SegmentationPicker().select(args.segment, partitions, win_length)
			explainer._segmenter = segmentation_class

			try:
				explaination, window = explainer.explain(segmentation_method=args.segpara) 
			except Exception as e:
				print(e)
				print('Error')
				explaination = None
				window = None 

			relevances.append(explaination.flatten())
			windows.append(window)
					
			print('Done:', i)

		seg_name= 'win_' + args.segment + '_' +args.segpara
		relevances_data.update({seg_name:windows,rel_res:relevances})

	with open(explain_name, 'wb') as f:
		dill.dump(relevances_data, f)



