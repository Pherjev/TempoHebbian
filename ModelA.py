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
print("Módulos importados...")

np.set_printoptions(precision=4)


K = 11
len_u = 2048 #1536 # 4096
W = np.zeros((K,len_u))

learned = 0

a0 = np.zeros((K,1))
diffA = np.zeros((K,1))
a = np.zeros((K+1,1))
hi= np.zeros((K+1,1))
h = np.zeros((K+1,1))
H = np.zeros((K+1,K+1))
A = np.ones((K,1))
C = np.ones((K+1,K+1))

print(h.shape)

for i in range(K+1):
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
			if x[i,0] > 75:
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

def Hebb(u,y,W):
        #theta = np.ones((K,1))
        tau = 0.5
        theta = 1.
        #u = X_train[i] # + np.random.rand(len_u)/10.
        u = u.reshape((len_u,1))
        #u = u / np.max(u)
        #print(u)
        v = np.zeros((K,1))
        v[int(y),0] = 1.

        #W += np.matmul(v - logistic(np.matmul(S(W),u)),u.T) # PERCEPTRON
        #W += np.matmul(v,u.T) # BASIC HEBB
        #theta += tau*(v**2 - theta)
        #W += np.matmul(v-0.4,u.T) # COVARIANCE
	#W += np.matmul(v,(u-10).T) # COVARIANCE 2
        W += np.matmul(v,u.T)-beta*((v**2)*np.ones((K,len_u)))*S(W) # OJA
        #W += np.matmul(v*(v-theta),u.T) # BCM
        #v2 = logistic(np.matmul(S(W),u))
        #W += np.matmul(v2*(v2-theta),u.T)
        
	return W



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

A00 = ''
t = 0

while(True):
	ret, frame = cap.read()
	
	# Feature extraction

	try:
		img = cv2.resize(frame,(299,299))
	except:
		img = np.zeros((299,299,3))


	frame = img_to_array(img)
	frame = frame.reshape((1, frame.shape[0], frame.shape[1], frame.shape[2]))
	frame = preprocess_input(frame)
	features = model.predict(frame, verbose=0)

	cv2.imshow('Omega',img)
	key = cv2.waitKey(1) & 0xFF
	#print(key)
	if key == ord('q'):
		break

	elif key == ord('r'):
		h[-1,0] = 1  

	elif key == ord('h'):
		print(H)
		print(A)
		break

	elif key == ord('s'):
		H = np.zeros((K+1,K+1))

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

				W = Hebb(features,idx,W)
				learned = 1

			else:
				print("Aumentar vocabulario:)")
		
		except:
			print("No escucho...:c")

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
	#DA = reward(hi,d,a)
	H += 0.05* ((np.matmul(h,hi.T)- np.matmul(hi,h.T))*C)
	H = control(H)
	#nmax = np.max(a0)
	#if nmax == 0:
	#	nmax = 1
	#A += 0.5*(h[0:11,:]*(a0 - 0.5)) + 0.05
	#print(A)
	#A = control(A)

	hi = 1*h

	t+= 1


print(A00)

# WEBGRAFIA

# https://towardsdatascience.com/convert-your-speech-to-text-using-python-1cf3eccfa922
