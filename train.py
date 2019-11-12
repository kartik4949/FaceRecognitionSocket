from face_recognition_final_v2 import Face_recognition


if __name__ == '__main__':
	fr = Face_recognition('face')
	#specify the directory containing faces!!
	fr.TrainFaces_SVM('Known_Faces/')

