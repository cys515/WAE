from curses import window
import matplotlib.pyplot as plt
from matplotlib.colorbar import ColorbarBase as cb
import numpy as np
import dill
import argparse
import sys
import math

def plotRes(args,ts_f,ts_len,seg_uniform,col,ts,idx2,metric,segment,explain,var_m):

	fig = plt.figure(figsize=(3.5, 4))

	ax0 = fig.add_axes([0.03, 0.675, 0.89, 0.26])	# (left, bottom, width, height)
	ax1 = fig.add_axes([0.03, 0.35, 0.89, 0.26])
	ax2 = fig.add_axes([0.03, 0.025, 0.89, 0.26])
	ax3 = fig.add_axes([0.94, 0.025, 0.03, 0.91])
	ax=[ax0,ax1,ax2,ax3]

	i = 0
	old_value = seg_uniform[0,0]
	for k in range(ts_f):
		for j in range(ts_len):
			value = seg_uniform[j,k]
			if old_value != value:
				old_value = value
				i += 1
			im = ax[k].axvline(x=j, linewidth=3.5, color=plt.get_cmap('Reds')(col[i]))

		ax[k].plot(ts[:,k],label='x'+str(k+1))
		ax[k].legend(loc='upper right',fontsize = 10)

	colorbar = cb(ax=ax[3],cmap =plt.get_cmap('Reds'),label=idx2,ticks = None)
	fig.text(0.5,-0.07,'Time')
	fig.text(-0.13,0.4,'Value',rotation='vertical')
	# plt.text(-0.1,-0.025,"var0="+str(format(var_m[0],'.4f')))
	# plt.text(-0.1,-0.05,"var1="+str(format(var_m[1],'.4f')))
	# plt.text(-0.1,-0.075,"var2="+str(format(var_m[2],'.4f')))
	# plt.text(-0.1,-0.1,"metric="+str(format(metric[0],'.4f')))
	fig.savefig("./Fig/"+args.dataName+str(args.step)+"_"+explain+"_"+segment+"_"+idx2+".pdf",bbox_inches='tight',pad_inches=0.1)
	plt.close()

def main(args):
	#load predictability and feature importance
	with open("./Results-len20/"+args.dataName+"_LSTM_"+str(args.step)+"_explain.dill", 'rb') as f:
		dataset = dill.load(f)

	#load raw data 
	with open('./Datasets/'+args.dataName+str(args.step)+'.dill', 'rb') as f1:
		data = dill.load(f1)
		ts = data[2]

	segment = ["window_uniform","matrix_slopes-not-sorted"]
	explain = ["LIME","LEMNA","SHAP"]
	# explain = ["SHAP"]

	for i in segment:
		# samples = list(map(lambda x: abs(x[0]-x[1]), zip(dataset['res_'+explain[0]+'_'+i], dataset['res_'+explain[1]+'_'+i])))
		# max_i = samples.index(max(samples))
		max_i = 0
		for j in explain:
			max_ss = dataset['res_'+j+'_'+i][max_i]
			window = dataset['win_'+i]
			color1 = dataset["res_"+j+'_'+i+"_pre"]
			color2 = dataset["res_"+j+'_'+i+"_IF"]

			ts_len = ts[max_i].shape[0]
			ts_f = ts[max_i].shape[1]

			metric = []
			tmp = 0
			tmp_window = np.unique(window[max_i].flatten()).tolist()
			tmp_window.sort()

			#calculate metric for every variable
			for k in range(len(tmp_window)):
				tmp = tmp + (color1[max_i][k] - color2[max_i][k])*(color1[max_i][k] - color2[max_i][k])
				tt=window[max_i][ts_len-1]
				if tmp_window[k] in tt:
					tmp=math.sqrt(tmp/(len(tmp_window)/ts_f))
					metric.append(tmp)
					tmp = 0
				
			# plotRes(args,ts_f,ts_len,window[max_i],color1[max_i],ts[max_i],"predictability",max_ss,i,j,metric)
			plotRes(args,ts_f,ts_len,window[max_i],color2[max_i],ts[max_i],"feature importance",max_ss,i,j,metric)

	# for j in explain:
	# 	samples = list(map(lambda x: abs(x[0]-x[1]), zip(dataset['res_'+j+'_'+segment[0]], dataset['res_'+j+'_'+segment[1]])))
	# 	max_i = samples.index(max(samples))
	# 	for i in segment:
	# 		max_ss = dataset['res_'+j+'_'+i][max_i]
	# 		window = dataset['win_'+i]
	# 		color1 = dataset["res_"+j+'_'+i+"_pre"]
	# 		color2 = dataset["res_"+j+'_'+i+"_IF"]

	# 		ts_len = ts[max_i].shape[0]
	# 		ts_f = ts[max_i].shape[1]

	# 		plotRes(args,ts_f,ts_len,window[max_i],color1[max_i],ts[max_i],"explain","predictability",max_ss,i,j)
	# 		plotRes(args,ts_f,ts_len,window[max_i],color2[max_i],ts[max_i],"explain","feature importance",max_ss,i,j)

def parse_arguments(argv):
	parser = argparse.ArgumentParser()
	#data para
	parser.add_argument('--dataName', type=str, default='P1')
	parser.add_argument('--step', type=int, default=60)
	return  parser.parse_args(argv)

if __name__ == '__main__':
	main(parse_arguments(sys.argv[1:]))
