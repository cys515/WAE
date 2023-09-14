import argparse
import dill
import sys 
from tensorflow import keras
import numpy as np
import random

def fid_cal(sample,win,IF,model):
		
		#original data prediction result
		orig = model.predict(sample[np.newaxis]).ravel()

		#mask important windows by explaination
		pert_data = np.copy(sample)
		explain = IF.tolist()
		win_num = np.unique(win)
		idx = dict(zip(win_num,explain))
		idx = sorted(idx.items(), key=lambda x: x[1], reverse=True)
		threshold = len(explain)//20
		pert_win = idx[0:threshold]
		for i in pert_win:
			idx = (win == i[0])
			pert_data[idx] = 0
		
		pert = model.predict(pert_data[np.newaxis]).ravel()

		#mask windows by random selecting
		rand_data = np.copy(sample)
		rand_win = random.sample(win_num.tolist(),threshold) #random select windows to mask
		for i in rand_win:
			idx = (win == i)
			rand_data[idx] = 0

		rand = model.predict(rand_data[np.newaxis]).ravel()

		np.seterr(divide='ignore', invalid='ignore')  # ingnore warning :divide=0
		score = abs((orig-rand)/(orig-pert))
		return score

def main(args):
	#load explaination
	with open("./Results/"+args.dataName+"_LSTM_300_explain.dill", 'rb') as f:
		data = dill.load(f)


	model_name='./Models/'+args.dataName+"_LSTM300"
	model = keras.models.load_model(model_name)
	
	win_type = {'matrix_slopes-not-sorted','window_uniform'}
	explain_type = {'LIME','LEMNA'}
	sample = data['samples']
			
	for i in win_type:
		for j in explain_type:
			print("********Evaluating "+args.dataName+" of "+j+" with "+i+"********")
			win = data['win_'+i]
			explain = data[j+'_'+i]
			fid = []
			for k in range(len(sample)):
				tmp = fid_cal(sample[k],win[k],abs(explain[k]),model)
				fid.append(tmp)

			data.update({'fid_'+j+'_'+i:fid})
	
	with open("./Results/"+args.dataName+"_LSTM_300_explain.dill", 'wb') as f:
		dill.dump(data, f)

def parse_arguments(argv):
	parser = argparse.ArgumentParser()
	#data para
	parser.add_argument('--dataName', type=str, default='P1')
	return  parser.parse_args(argv)

if __name__ == '__main__':
	main(parse_arguments(sys.argv[1:])) 
