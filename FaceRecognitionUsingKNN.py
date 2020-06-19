import numpy as np
import os
import cv2

def collectData():
	cam=cv2.VideoCapture(0) #creation of object which links to webcam
	faceCascade=cv2.CascadeClassifier("https://github.com/opencv/opencv/raw/master/data/haarcascades/haarcascade_frontalface_default.xml") #creation of an object which contains imortant function detectMultiScale

	faceData=[]
	dataSetPath="./faceDataFile/"

	name=input("Enter your name: ")

	while len(faceData)<=1000:
		ret,frame=cam.read()
		if ret==False:
			continue
		gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) #it converts bgr image to grayscale because detectMultiScale always accepts 
		faces=faceCascade.detectMultiScale(gray_frame,1.3,5) #1.3 is the scale with which image is to be resized and 5 is minimum number of neighbors required or minimum no. of faces/glows detected at that point to be considered it as a face

		for face in faces: #because I don't want that the another person add to my face data
			x,y,w,h=face
			cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2) #to make a rectangle around the face inside frame
			offset=10
			faceSection=frame[y-offset:y+h+offset,x-offset:x+w+offset]
			faceSection=cv2.resize(faceSection,(100,100))
			faceData.append(faceSection)
			cv2.imshow("Cropped image",faceSection)
			cv2.putText(frame,"Collecting face data....",(x,y-offset),cv2.FONT_HERSHEY_PLAIN,1,(255,255,255),1,cv2.LINE_AA)
		cv2.imshow("Window",frame) #to print the frame with name window
		key=cv2.waitKey(1) #to hold the screen having frame
		if key==ord("q"):
			break

	print(f"{len(faceData)} images collected.")
	#print(faceData[0].shape)

	faceData=np.array(faceData)
	faceData=faceData.reshape(faceData.shape[0],-1)
	#print(faceData.shape)
	np.save(dataSetPath+name+".npy",faceData)
	print("Face data saved at "+dataSetPath+name+".npy")

	cam.release()
	cv2.destroyAllWindows()



def recogFace():
	dataSetPath="./faceDataFile/"

	# Merge the face data of all person
	faceData=[]
	labels=[]

	for file in os.listdir(dataSetPath):
		if file.endswith(".npy"):
			label=file.split(".")[0]
			fileData=np.load(dataSetPath+file)

			#print(fileData.shape)
			#print("Face data of "+label+" has been loaded")

			faceData.append(fileData)
			for i in range(fileData.shape[0]):
				labels.append(label)


	X=np.concatenate(faceData,axis=0)
	Y=np.array(labels)

	#print(X.shape)
	#print(Y.shape)

	#kNN code here....

	def distance(pA,pB):
		return np.sum((pA-pB)**2)**0.5

	def kNN(X,Y,query,k=5):
		"""
		X --> (m,30000) np-array
		Y --> (m,) np-array
		query --> (30000,) np-array
		k --> count of nearest neighbours to be considered

		This kNN is for Classification
		"""

		m=X.shape[0]

		distances=[]
		for i in range(m):
			dis=distance(X[i],query)
			distances.append((dis,Y[i]))

		distances=sorted(distances)
		distances=distances[:k]
		distances=np.array(distances)
		labels=distances[:,1]
		uniq_label,count=np.unique(labels,return_counts=True)
		pred=uniq_label[count.argmax()]

		return pred


	#Test face recognition

	cam=cv2.VideoCapture(0) #creation of object which links to webcam
	faceCascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml") #creation of an object which contains imortant function detectMultiScale

	while True:
		ret,frame=cam.read()
		if ret==False:
			continue
		gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) #it converts bgr image to grayscale because detectMultiScale always accepts 
		faces=faceCascade.detectMultiScale(gray_frame,1.3,5) #1.3 is the scale with which image is to be resized and 5 is minimum number of neighbors required or minimum no. of faces/glows detected at that point to be considered it as a face

		for face in faces: #because I don't want that the another person add to my face data
			x,y,w,h=face
			cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2) #to make a rectangle around the face inside frame
			offset=10
			faceSection=frame[y-offset:y+h+offset,x-offset:x+w+offset]
			faceSection=cv2.resize(faceSection,(100,100))
			
			findName=kNN(X,Y,faceSection.reshape(1,-1))

			cv2.putText(frame,findName.title(),(x,y-offset),cv2.FONT_HERSHEY_PLAIN,1,(153,255,255),1,cv2.LINE_AA)

		cv2.imshow("Window",frame) #to print the frame with name window
		key=cv2.waitKey(1) #to hold the screen having frame
		if key==ord("q"):
			break

	cam.release()
	cv2.destroyAllWindows()

print("Are you an existing user or a new user?")
reply=input('If existing user then print "exist" otherwise "new" : ')
if reply=="new":
	collectData()
	print("Your data has been successfully collected.Now start again as an existing user...")
elif reply=="exist":
	recogFace()
else:
	print("Invalid input....")