
import numpy as np
from scipy.sparse import csc_matrix,eye,diags
from scipy.sparse.linalg import spsolve

def WhittakerSmooth(x,w,lambda_):
	'''

	:param x: array 输入数值，数组
	:param w: array 与数值对应的权重数组
	:param lambda_: 平滑参数
	:return: array 平滑结果
	'''
	X=np.matrix(x)#这里将数组转化为矩阵。矩阵之后就不可以用索引进行引用了。
	m=X.size
	#i=np.arange(0,m)
	E=eye(m,format='csc')
	D=E[1:]-E[:-1] # numpy.diff() does not work with sparse matrix. This is a workaround.
	W=diags(w,0,shape=(m,m))
	A=csc_matrix(W+(lambda_*D.T*D))
	B=csc_matrix(W*X.T)
	background=spsolve(A,B)   #求解矩阵方程

	return np.array(background)

def TD_baseline(time,rate,lam = None,hwi = None,it = None,inti = None):
	dt = time[1]-time[0]

	if(lam is None):
		lam = 100/dt
	if(hwi is None):
		hwi = int(20/dt)
	if(it is None):
		it = 5
	if(inti is None):

		fillpeak_int = int(len(rate)/10)

	else:
		fillpeak_int =inti
	if(lam < 1):
		lam = 1
	bs = baseline(rate,lambda_=lam,hwi=hwi,it = it,int_ = fillpeak_int)
	return time,rate-bs,bs


def get_smooth(spectra,lambda_):
	spectra = np.array(spectra)
	m = spectra.shape[0]
	w = np.ones(m)
	smooth = WhittakerSmooth(spectra,w,lambda_)
	cs = spectra-smooth
	cs_mean = cs.mean()
	cs_std = cs.std()
	for i in range(3):
		cs_index = np.where((cs>cs_mean+(1+1*i)*cs_std)|(cs<cs_mean-(1+1*i)*cs_std))
		w[cs_index] = 0
		smooth = WhittakerSmooth(spectra,w,lambda_)
		cs = spectra-smooth
		cs_mean = cs[w!=0].mean()
		cs_std = cs[w!=0].std()
	return smooth


def baseline(spectra,lambda_,hwi,it,int_):
	spectra = np.array(spectra)
	spectra = get_smooth(spectra,lambda_)

	if it != 1 :
		d1 = np.log10(hwi)
		d2 = 0
		w = np.ceil(np.concatenate((10**(d1+np.arange(0,it-1,1)*(d2-d1)/(np.floor(it)-1)),[d2])))
		w = np.array(w,dtype = int)
	else:
		w = np.array([hwi],dtype = int)
	#print(w)

	lims = np.linspace(0,spectra.size -1,int_+1)
	lefts = np.array(np.ceil(lims[:-1]),dtype = int)#这里指的是索引值
	rights = np.array(np.floor(lims[1:]),dtype = int)#同上
	minip = (lefts+rights)*0.5#索引
	xx = np.zeros(int_)
	for i in range(int_):#这里是一个rebin的过程,这里可以提速
		xx[i] = spectra[lefts[i]:rights[i]+1].mean()

	'''
	spectra_index = np.arange(0, spectra.size, 1)
	mreg_n = 10 #int(spectra.size/int_)
	cutoff = int(spectra.size/mreg_n)*mreg_n
	spectra_A = spectra[:cutoff]
	spectra_index_A = spectra_index[:cutoff]
	xx_reshape = spectra_A.reshape(-1,mreg_n)
	xx = xx_reshape.mean(axis = 1)
	xx_index_reshape = spectra_index_A.reshape(-1,mreg_n)
	minip = xx_index_reshape.mean(axis = 1)
	if(spectra.size % mreg_n !=0):
		xx_b = spectra[cutoff:]
		xx_index_b = spectra_index[cutoff:]
		xx = np.concatenate((xx,[xx_b.mean()]))
		minip = np.concatenate((minip,[xx_index_b.mean()]))
	int_ = xx.size
	'''
	for i in range(it):
		# Current window width
		w0 = w[i]
		# Point-wise iteration to the right
		for j in range(1,int_-1):
			# Interval cut-off close to edges
			v = min([j,w0,int_-j-1])
			# Baseline suppression
			a = xx[j-v:j+v+1].mean()
			xx[j] = min([a,xx[j]])
		for j in range(1,int_-1):
			k = int_-j-1
			v = min([j,w0,int_-j-1])
			a = xx[k-v:k+v+1].mean()
			xx[k] = min([a,xx[k]])

	minip = np.concatenate(([0],minip,[spectra.size-1]))
	xx = np.concatenate((xx[:1],xx,xx[-1:]))
	index = np.arange(0,spectra.size,1)
	xxx = np.interp(index,minip,xx)
	return xxx
























