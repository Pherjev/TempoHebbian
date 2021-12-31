#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import speech_recognition as sr
from tensorflow.keras.applications import Xception
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Model
from tensorflow.keras.applications.xception import preprocess_input
from os import system
import time
print("Módulos importados...")

np.set_printoptions(precision=4)

umbral = 70

K = 11
Q = 5
len_u = 2048 #1536 # 4096
W = np.zeros((K,len_u))

learned = 0

theta = np.zeros((K,1))

a0 = np.zeros((K,1))
diffA = np.zeros((K,1))
a = np.zeros((K,1))
v = np.zeros((K,1))
hi= np.zeros((K,1))
h = np.zeros((K,1))
H = np.zeros((K,K))
A = np.ones((K,1))
C = np.ones((K,K))


P = np.zeros((K,10))


for i in range(K):
	P[i,0] = 1


# Dopamine System

r = 0
VW = np.zeros((K,10))
x1 = np.zeros((K,10))
x2 = np.zeros((K,10))
M = np.zeros((10,10))

for i in range(len(M)-1):
	M[i+1,i] = 1.


#print(h.shape)

for i in range(K):
	C[i,i] = 0
	H[i,i] = 0.8


beta = 0.01
vocabulario = []

def reconocer():
	r = sr.Recognizer()
	mic = sr.Microphone(device_index=0)
	with mic as source:
		audio = r.listen(source)
	result = r.recognize_google(audio,language="es")
	return result

def softmax(x):
	return x/(np.max(x)) #np.exp(x)/float(np.sum(np.exp(x)))

print("Iniciando...")


print("Prueba de audio...")
try:
	print(reconocer()) # BORRAR
except:
	print("ERROR")

def S(W): # Activacion
        n,m = W.shape
        L = W >= 1
        L2= W < 1
        B = W*L
        O = np.ones((n,m))
        C = B + O*(W*L == 0.)
        D = np.log(C) + L*O
        D += W*L2
        return D

def logistic(W):
        return 1./(1.+np.exp(-W))

def control(W):
	L0 = W > 1
	L1 = W <= 1
	L2 = W >= 0
	L = L1*L2
	W = W*L + L0 
	return W

def activacion(x, mode = 'hardlims'):
	if mode == 'hardlims':
		for i in range(len(x)):
			if x[i,0] > 1:
				x[i,0] = 1
			elif x[i,0] < 0.1:
				x[i,0] = 0
		return x
	elif mode == 'argmax':
		maxi = np.argmax(x)
		for i in range(len(x)):
			if i == maxi and x[i,0] > 0:
				x[i,0] = 1
			else:
				x[i,0] = 0
		return x
	elif mode == 'umbral':
		for i in range(len(x)):
			if x[i,0] > umbral:
				x[i,0] = 1
			else:
				x[i,0] = 0
		return x
	else:
		return x

def dynamics(x,W,repeat = 1):
  for epocs in range(repeat):
    x = activacion(np.matmul(W,x))
  return x


def Hebb(u,v,W,theta=theta):
        #theta = np.ones((K,1))
        tau = 0.5
        #theta = 1.
        #u = X_train[i] # + np.random.rand(len_u)/10.
        u = u.reshape((len_u,1))
        #u = u / np.max(u)
        #print(u)
        #v = np.zeros((K,1))
        #v[int(y),0] = 1.

        #W += np.matmul(v - logistic(np.matmul(S(W),u)),u.T) # PERCEPTRON
        #W += np.matmul(v,u.T) # BASIC HEBB
        theta += tau*(v**2 - theta)
        #W += np.matmul(v-0.4,u.T) # COVARIANCE
	#W += np.matmul(v,(u-10).T) # COVARIANCE 2
        #W += np.matmul(v,u.T)-beta*((v**2)*np.ones((K,len_u)))*S(W) # OJA
        W += np.matmul(v*(v-theta),u.T) # BCM
        #v2 = logistic(np.matmul(S(W),u))
        #W += np.matmul(v2*(v2-theta),u.T)
        
	return W

def reward(h1,d,a):
	DA = np.zeros((K,K))
	for i in range(K):
		for j in range(K):
			if i == j:
				DA[i,j] = 1
			else:
				if a[j,0] >= 0.5 and h1[i,0] < 0.5:
					if d[j,0] >= 0.5: # Most correctly predicted reward
						DA[j,i] = 0.5
					else: # Unexpected reward
						DA[j,i] = 5
				elif a[j,0] >= 0.5 and h1[i,0] >= 0.5:
					if d[j,0] >= 0.5: # Correctly predicted reward
						DA[j,i] = 0.1
					else: # Mostly unexpected reward
						DA[j,i] = 10
				elif a[j,0] < 0.5 and h1[i,0] < 0.5:
					if d[j,0] >= 0.5: # Most incorrectly predicted reward
						DA[j,i] = -1 
					else:  # Correctly predicted absence of reward
						DA[j,i] = -0.25
				elif a[j,0] < 0.5 and h1[i,0] >= 0.5:
					if d[j,0] >= 0.5: # Incorrectly predicted reward
						DA[j,i] = -0.8
					else:  # Correctly predicted absence of reward
						DA[j,i] = -0.2
	return DA


def clasificacion(u,W):
	u = u.reshape((len_u,1))
	v = np.matmul(S(W),u)
	v = v.reshape(K)
	v = np.argmax(v)
	return v

def clasificacion2(u,W):
        u = u.reshape((len_u,1))
        v = np.matmul(S(W),u)
        #v = v.reshape(K)
        return v

def rescale_frame(frame, percent=75):
	width = int(frame.shape[1] * percent/ 100)
	height = int(frame.shape[0] * percent/ 100)
	dim = (width, height)
	return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)


model = Xception()
model.layers.pop()
model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
model.summary()

system("say ¡Hola!")

print("Hola c:")

print()
print("______________________________")
print("Controles:")
print("Tecla q: Salir")
print("Tecla r: Recompensa")
print("Tecla a: Escuchar")
print("Tecla b: Revisar vocabulario")
print("Tecla c: Reconocer")
print("______________________________")

cap = cv2.VideoCapture(0)
#cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
#width = 640
#height = 480
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
#cap.set(5,60)
#cap.set(3, 640)
#cap.set(4, 480)


A00 = ''
B00 = ''
C00 = ''
D00 = ''


A10 = []
B10 = []
C10 = []
D10 = []
A11 = []
B11 = []
C11 = []
D11 = []
Z11 = []
R11 = []


t = 0

t0 = time.time()

while(True):
	ret, frame = cap.read()
	
	#frame = rescale_frame(frame, percent=25)

	# Feature extraction

	try:
		#print(frame.shape)
		#(720, 1280, 3)
		img= frame[210:509,490:789,:]#[60:660,340:940,:]
		#print(frame.shape)
		#img = cv2.resize(frame,(299,299))
	except:
		img = np.zeros((299,299,3))


	frame = img_to_array(img)
	frame = frame.reshape((1, frame.shape[0], frame.shape[1], frame.shape[2]))
	frame = preprocess_input(frame)
	features = model.predict(frame, verbose=0)
	features = softmax(features)
	#print(features)

	cv2.imshow('Omega',img)
	key = cv2.waitKey(1) & 0xFF
	#print(key)
	if key == ord('q'):
		break

	elif key == ord('r'):
		r = 1  

	elif key == ord('h'):
		print(H)
		print(A)
		break

	elif key == ord('s'):
		H = np.zeros((K,K))

	elif key == ord('f'):
		print('Training complete...')
		t = 0
		A00 = ''
		B00 = ''
		C00 = ''
		D00 = ''

	elif key == ord('a'):
		print("Escuchando...")

		try:
			R = reconocer()
			print(R)		
			system('say ' + R)
			idx = 0

			if len(vocabulario) < K:
				if R in vocabulario:
					idx = vocabulario.index(R)		
				else:
					vocabulario.append(R)
					idx = vocabulario.index(R) #len(vocabulario) - 1
				
				v[idx,0] = 1.
				W = Hebb(features,v,W)
				#print(W)
				learned = 1

			else:
				print("Aumentar vocabulario:)")
		
		except:
			print("No escucho...:c")
		#W = Hebb(features,idx,W)

	elif key == ord('b'):
		print("Vocabulario:")
		print(vocabulario)

	elif key == ord('c'):
                print("Reconocimiento...")
                c = clasificacion(features,W)
		print(vocabulario[c])
		system('say ' + vocabulario[c])

	
	# Red Secundaria

	a0 =  clasificacion2(features,W)*1
	A00+= '('+str(t) + ',' +  str(a0[0,0]) + ')'
	B00+= '('+str(t) + ',' +  str(a0[1,0]) + ')'
	C00+= '('+str(t) + ',' +  str(a0[2,0]) + ')'
	D00+= '('+str(t) + ',' +  str(a0[3,0]) + ')'

	A10.append(a0[0,0])
	B10.append(a0[1,0])
	C10.append(a0[2,0])
	D10.append(a0[3,0])
	

	print("__________________")
	print(a0)
	a0 = np.reshape(a0,(-1,1))
	a0 = activacion(a0,mode='umbral')

	a[0:K] = A*a0
        #print("__________________")
        #print(a)
	#h = np.reshape(h,(-1,1))
	#print(h)
	#print(h.shape,H.shape)
	d = dynamics(h,H)
	#d = np.reshape(d,(-1,1))
	h = activacion(a + d)
	print()
	print(h.T)


	A11.append(h[0,0])
	B11.append(h[1,0])
	C11.append(h[2,0])
	D11.append(h[3,0])


	#DA = reward(hi,d,a)

	#da = a-d
	#H += 0.05*((np.matmul(h,da.T)- np.matmul(da,h.T))*C)


	H += 0.04* ((np.matmul(h,hi.T)- np.matmul(hi,h.T))*C)
	H = control(H)
	#nmax = np.max(a0)
	#if nmax == 0:
	#	nmax = 1
	#A += 0.5*(h[0:11,:]*(a0 - 0.5)) + 0.05
	#print(A)
	#A = control(A)

	# Dopamine System

	z2 = 0

	for i in range(K):
		#x2[i,0] = h[i,0]
		x2[i,:] = P[i,:]*h[i]
		#df = x2[i,:] - x1[i,:]
		y = np.max(x2[i])  #np.matmul(VW[i].T,df)

		print(y)
		print(Lizzy)


		# Sacar z para generalizar

		z = y + r
		#VW[i,:] += x1[i,:]*z

		P[i,:] += x1[i,:]*z

		z2 += z

		#if z >= 1:
		#	print("REWARD")
		
		#print(z)

		x1[i,:] = x2[i,:]*1
		x2[i,:] = np.matmul(M,x2[i,:])
	
	print('z = ',z2)
	if z2 >= 1:
		print("REWARD")



	hi = 1*h
	v = activacion(0.5*v)
	
	Z11.append(z2)
	R11.append(r)


	r = 0

	t+= 1


#print(A00)
#print()
#print(B00)
#print()
#print(C00)
#print()
#print(D00)



t1 = time.time()

print('FPS=',t/(t1-t0))

def guardar(titulo,var):
	var = np.array(var)
        with open(titulo, 'wb') as f:
                np.save(f, var)



guardar('RingModelC_v1.npy',A10)
guardar('RingModelC_v2.npy',B10)
guardar('RingModelC_v3.npy',C10)
guardar('RingModelC_v4.npy',D10)
guardar('RingModelC_r1.npy',A11)
guardar('RingModelC_r2.npy',B11)
guardar('RingModelC_r3.npy',C11)
guardar('RingModelC_r4.npy',D11)
guardar('RingModelC_z.npy',Z11)
guardar('RingModelC_r.npy',R11)



#print(theta)
#print(W)

# WEBGRAFIA

# https://towardsdatascience.com/convert-your-speech-to-text-using-python-1cf3eccfa922
