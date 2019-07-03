import os
import numpy as np
import params as p
import matplotlib.pyplot as plt
import time
import matplotlib.animation as animation

def plotmap(worldmap):
	for i in range(p.a):
		for j in range(p.b):
			if worldmap[i,j]>0:
				plt.scatter(i,j,color='black')
	plt.show()

def Qlambda(w,state):
	f=featfromstate(state)
	e=np.zeros((len(f),p.A))
	###loaded_files=np.load('saved_data.npy.npz')
	###wopt=(loaded_files['arr_0'])
	count=0
	next_state=state.copy()
	err=0
	f=featfromstate(state)
	a=np.random.randint(p.A)
	ef=e[:,a]
	breakflag=0
	while np.linalg.norm(state-p.targ)>p.thresh:
		count=count+1
		if breakflag==1:
			break
		if count>p.breakthresh:
			breakflag=1
			print('Broke')
		#epsilon greedy exploration
		if p.epsilon>np.random.sample():
			#explore
			a=np.random.randint(p.A)
		else:
			#exploit
			Qmax,a=maxQ(w,state)
		#define reward structure
		next_state=transition(state,a)
		#time.sleep(5)
		if np.linalg.norm(next_state-p.targ)<=p.thresh:
			R=p.highreward
		elif p.world[int(np.around(next_state[0])),int(np.around(next_state[1]))]==1:
			R=p.penalty
		else:
			R=p.livingpenalty
		###Qmaxmeannext=meanQ(wopt,next_state)
		###R=0
		###R=Q_s_a(wopt,state,a)-p.gamma*Qmaxmeannext#inferred reward from Q*
		#shaping=p.gamma*(100-np.linalg.norm(next_state-p.targ))-(100-np.linalg.norm(state-p.targ))
		#vs=Qmaxmean
		#vss=Qmaxmeannext+++++++
		#shaping=vss-vs
		err=R-Q_s_a(w,state,a)#+shaping
		#update w
		f=featfromstate(state)
		Qmaxopt,optact=maxQ(w,state)
		if a!=optact:
			e=np.zeros((len(f),p.A))
		ef=e[:,a]
		for i in range(len(f)):
			if f[i]==1:
				ef[i]=1
		#print(next_state)
		#if R==p.highreward:
		if np.linalg.norm(next_state-p.targ)<=p.thresh:
			w[:,a]=w[:,a]+p.alpha*err*ef
			breakflag=1
		else:
			Qmax_next, a_next=maxQ(w,next_state)
			err=err+p.gamma*Qmax_next
			w[:,a]=w[:,a]+p.alpha*err*ef
			e[:,a]=ef.copy()
			e=p.lambd*p.gamma*e
		state=next_state.copy()
	return w



def transition(state,act):
	#print(orig_state)
	#print(act)
	new_state=state.copy()
	if act==0:
		new_state[0]=state[0]
		new_state[1]=state[1]+p.v*p.dt#move up
	elif act==1:
		new_state[0]=state[0]+p.v*p.dt#right
		new_state[1]=state[1]
	elif act==2:
		new_state[0]=state[0]
		new_state[1]=state[1]-p.v*p.dt#move down
	elif act==3:
		new_state[0]=state[0]-p.v*p.dt#move left
		new_state[1]=state[1]
	elif act==4:
		new_state[1]=state[1]+p.v*p.dt#move diagonally up,left
		new_state[0]=state[0]-p.v*p.dt
	elif act==5:
		new_state[1]=state[1]+p.v*p.dt#move diagonally up,right
		new_state[0]=state[0]+p.v*p.dt
	elif act==6:
		new_state[1]=state[1]-p.v*p.dt#move diagonally down,right
		new_state[0]=state[0]+p.v*p.dt
	elif act==7:
		new_state[1]=state[1]-p.v*p.dt#move diagonally down,left
		new_state[0]=state[0]-p.v*p.dt
	elif act==8:
		new_state=state.copy()

	if new_state[0]>=p.a-1 or new_state[0]<=0 or new_state[1]>=p.b-1 or new_state[1]<=0 or p.world[int(np.around(new_state[0])),int(np.around(new_state[1]))]==1:
		new_state=state.copy()
	#new_state=state[:]
	#print(state)
	return new_state



def maxQ(w,state):
	Q=[]
	for i in range(p.A):
		Q.append(Q_s_a(w,state,i))
	Qmax=np.max(Q)
	a=np.argmax(Q)
	return Qmax,a

def meanQ(w,state):
	Q=[]
	for i in range(p.A):
		Q.append(Q_s_a(w,state,i))
	Qmean=np.max(Q)
	return Qmean

def Q_s_a(w,state,action):
	f=featfromstate(state)
	w_act=w[:,action]
	Q=np.dot(w_act,f)
	return Q

def featfromstate(state):
	f1=np.zeros((p.nfeatsx,1))
	f1[int(np.around(p.nfeatsx*(state[0]/p.a)))]=1
	f2=np.zeros((p.nfeatsy,1))
	f2[int(np.around(p.nfeatsy*(state[1]/p.b)))]=1
	f=np.concatenate((f1,f2),axis=0)
	return f

def Qlearn_main_vid():
	Qimall=[]
	state=(p.a-1)*np.random.random_sample(2)
	f=featfromstate(state)
	w = np.random.random_sample((len(f), p.A))
	for i in range(p.episodes):
		print(i)
		initial_state=(p.a-1)*np.random.random_sample(2)
		w=Qlambda(w,initial_state)
		Qmap,Qfig=mapQ_vid(w)
		Qimall.append([Qfig])
		fig = plt.figure("Qmap_video")
		ani = animation.ArtistAnimation(fig, Qimall, interval=20, blit=True,repeat_delay=0)
		Writer = animation.writers['ffmpeg']
		writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
		ani.save('Qmapvid.mp4', writer=writer)
	return w,Qimall

def Qlearn_main():
	state=(p.a-1)*np.random.random_sample(2)
	f=featfromstate(state)
	w = np.random.random_sample((len(f), p.A))
	#loaded_files=np.load('saved_data.npy.npz')
	#wopt=(loaded_files['arr_0'])
	#w=wopt.copy()
	returns=[]
	for i in range(p.episodes):
		if (i+1)/p.episodes==0.25:
			print('25% episodes done')
		elif (i+1)/p.episodes==0.5:
			print('50% episodes done')
		elif (i+1)/p.episodes==0.75:
			print('75% episodes done')
		elif (i+1)/p.episodes==1:
			print('100% episodes done')
		if i%10==0:
			returns.append(calcret(w))
		initial_state=(p.a-1)*np.random.random_sample(2)
		w=Qlambda(w,initial_state)
		#Qmap=mapQ(w)
	return w,returns

def opt_pol(w,state):
		plt.figure(0)
		plt.ion()
		for i in range(p.a):
			for j in range(p.b):
				if p.world[i,j]>0:
					plt.scatter(i,j,color='black')
		plt.show()
		pol=[]
		statelog=[]
		count=1
		while np.linalg.norm(state-p.targ)>=p.thresh:
			Qm,a=maxQ(w,state)
			if np.random.sample()>0.9:
				a=np.random.randint(p.A)
			next_state=transition(state,a)
			pol.append(a)
			statelog.append(state)
			print(state)
			plt.ylim(0, p.b)
			plt.xlim(0, p.a)
			plt.scatter(state[0],state[1],color='blue')
			plt.draw()
			plt.pause(0.3)
			#input("Press [enter] to continue.")
			state=next_state.copy()
			print(count)
			if count>=100:
				break
			count=count+1
		return statelog,pol


def mapQ(w):
	plt.figure(1)
	plt.ion
	Qmap=np.zeros((p.a,p.b))
	for i in range(p.a):
		for j in range(p.b):
 			Qav=0
 			for k in range(p.A):
 				Qav=Qav+Q_s_a(w,[i,j],k)
 			Qmap[i,j]=Qav
	#Qfig=plt.imshow(np.rot90(Qmap))
	plt.imshow(np.rot90(Qmap),cmap="gray")
	#plt.draw()
	plt.pause(0.0001)
	return Qmap

def mapQ_vid(w):
	Qmap=np.zeros((p.a,p.b))
	for i in range(p.a):
		for j in range(p.b):
 			Qav=0
 			for k in range(p.A):
 				Qav=Qav+Q_s_a(w,[i,j],k)
 			Qmap[i,j]=Qav
	Qfig=plt.imshow(np.rot90(Qmap),cmap="gray")
	return Qmap,Qfig

def calcret(w):
	ret=0
	for i in range(p.evalruns):
		state=(p.a-1)*np.random.random_sample(2)
		for j in range(p.evalsteps):
			if np.linalg.norm(state-p.targ)<=p.thresh:
				R=p.highreward
			elif p.world[int(np.around(state[0])),int(np.around(state[1]))]==1:
				R=p.penalty
			else:
				R=p.livingpenalty
			Qmaxopt,optact=maxQ(w,state)
			state=transition(state,optact)
			ret=ret+R
	avgsumofrew=ret/p.evalruns
	return avgsumofrew


def Qlearn_multirun():
	retlog=[]
	for i in range(p.Nruns):
		w,ret=Qlearn_main()
		if i==0:
			retlog=ret
		else:
			retlog=np.vstack((retlog,ret))
		#retlog.append(ret)
		if (i+1)/p.Nruns==0.25:
			print('25% runs complete')
		elif (i+1)/p.Nruns==0.5:
			print('50% runs complete')
		elif (i+1)/p.Nruns==0.75:
			print('75% runs complete')
		elif (i+1)==p.Nruns:
			print('100% runs complete')
	meanreturns=(np.mean(retlog,axis=0))
	plt.plot(meanreturns)
	plt.show()
	return w, retlog


#######################################
if __name__=="__main__":
	#w,Qimall=Qlearn_main_vid()
	w,retlog=Qlearn_multirun()

########Saving and loading files##########
#np.savez('saved_data.npy',w,retlog)
#loaded_files=np.load('saved_data.npy.npz')
#loaded_files.files
#loaded_files['arr_0']
#w=Qlearn_main()

######Figures########
#plt.savefig('destination_path.eps', format='eps', dpi=1000)

######optimal policy#####
#statelog,pol=opt_pol(w,np.array([15.,25.]))
