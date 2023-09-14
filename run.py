import argparse
import sys
import os
from createDatasets import createDill
from train_models import train_models
from explain import explain

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

def main(args):

	# Creating Datasets
	Datafile='./Datasets/'+args.DataName
	if not os.path.exists(Datafile+str(args.step)+'.dill'):
		createDill(args,Datafile)


	#training model
	model_name='./Models/'+args.DataName+"_"+args.model + str(args.step) 
	if not os.path.exists(model_name):
		train_models(args,Datafile+str(args.step)+'.dill',model_name)

	#create explain
	explain(args,Datafile+str(args.step)+'.dill',model_name)


def parse_arguments(argv):
	parser = argparse.ArgumentParser()
	#data para
	parser.add_argument('--DataName', type=str, default='pufa', help='name of the dataset [ Test | Gx ]')
	parser.add_argument('--features', type=int, default=3, help='numbers of feature in multivariable time series')
	parser.add_argument('--times', type=int, default=1000, help='time steps of multivariable time series')
	parser.add_argument('--correlation',type=int,default=0,help='the correlation between features [ 0(independent) | 1(half-dependent) | 2(dependent) ]')

	#model para
	parser.add_argument('--step', type=int, default=60, help='lag for pridiction')
	parser.add_argument('--model', type=str, default="LSTM", help='name of black model [LSTM]')

	#explain para
	parser.add_argument('--explain', type=str, default="SHAP", help='name of explaining models [LIME | LIMNA |SHAP]')
	parser.add_argument('--segment', type=str, default="matrix", help='type of segment methods [window | matrix | sax]')
	parser.add_argument('--segpara', type=str, default="slopes-not-sorted", help='type of segment methods corresponding to --segment  [window:[uniform | exponential] | matrix:[slopes-sorted |slopes-not-sorted |bins-max| bins-min]]} | sax]')
	return  parser.parse_args(argv)

if __name__ == '__main__':
	main(parse_arguments(sys.argv[1:])) 
