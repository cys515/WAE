from cmath import nan
import dill
import sys
import argparse
from sympy import log,nsolve,Symbol,re
import numpy as np
import math 

def if_contain(A,B):
	n = len(A)
	return any(A == B[i:i + n] for i in range(len(B)-n + 1))
 

def cal_entropy(win,sample):

	#get time series
	sample1 = sample.T.ravel()
	size = (np.max(sample1)-np.min(sample1))/10
	sample2 = [(int)(item/size) for item in sample1] 
	win1 = list(map(int,win.T.ravel()))
	windows = np.unique(win1)
	ts = []
	for i in windows:
		ts_tmp = []
		for index, value in enumerate(win1):
			if value == i:
				ts_tmp.append(sample2[index])
		ts.append(ts_tmp)

	#LZentropy
	pre = []
	for t in ts:
		l = len(t)
		N = len(list(set(t)))
		sum_len = 0
		flag = True
		prefix_list = []
		for i in range(l):
			if(flag):
				prefix = [t[i]]
				k=1
				while(if_contain(prefix,prefix_list) and (i+k)<l):
					prefix.append(t[i+k])
					k = k+1
				
				sum_len = sum_len+k
				prefix_list.append(t[i])
				if(i+k)>=l:
					flag = False
			else:
				break
		if l<2:
			S = 0
		else:
			S = l*(math.log(l,2))/sum_len

		#calculate predictability
		p = Symbol('p',real=True)
		if N>1:
			p = float(re(nsolve(-p*log(p,2)-(1-p)*log((1-p),2)+(1-p)*log((N-1),2)-S,p,0.5,verify=False)))
		else: 
			p = 0
		pre.append(p)
	
	return pre 

def main(args):
	#load explaination
	with open(args.dataName, 'rb') as f:
		data = dill.load(f)
	
	win_type = {'matrix_slopes-not-sorted','window_uniform'}

	for i in win_type:
		print("********Evaluating "+args.dataName+" of "+args.explain+" with "+i+"********")
		sample = data['samples']
		win = data['win_'+i]
		explain = data[args.explain+'_'+i]
		metric = []
		pre = []
		IF = []
		for j in range(len(explain)):

			imp_factor = np.expand_dims(abs(explain[j]),axis=1)
			imp_factor = imp_factor/imp_factor.sum()
			IF.append(imp_factor)

			p = cal_entropy(win[j],sample[j])
			p = np.expand_dims(p,axis=1).real
			win_tmp = win[j][-1]
			tmp_window = np.unique(win[j].flatten()).tolist()
			tmp_window.sort()
			win_no1 = -1
			for k in win_tmp:
				win_no = tmp_window.index(k)
				win_count = win_no - win_no1
				win_no1 = win_no
				ss = 0
				for m in range(win_count):
					ss = ss + m + 1
				for m in range(win_count):
					p[win_no-win_count+m+1] = abs(p[win_no-win_count+m+1])*(m+1)/ss

			p = p/p.sum()
			pre.append(p)

			#res = p * imp_factor
			#res = np.corrcoef(p.T, imp_factor.T)[0,1]

			res = (p - imp_factor)*(p - imp_factor)

			metric.append(sum(res)/len(res))
		data.update({'res_'+args.explain+'_'+i:metric})
		data.update({'res_'+args.explain+'_'+i+'_IF':IF})
		data.update({'res_'+args.explain+'_'+i+'_pre':pre})
	
	with open(args.dataName, 'wb') as f:
		dill.dump(data, f)

def parse_arguments(argv):
	parser = argparse.ArgumentParser()
	#data para
	parser.add_argument('--dataName', type=str, default='./Results/pufa_LSTM_60_explain.dill')
	parser.add_argument('--explain', type=str, default='LEMNA', help='name of explaining models [LIME | LEMNA |SHAP]')
	return  parser.parse_args(argv)

if __name__ == '__main__':
	main(parse_arguments(sys.argv[1:])) 
