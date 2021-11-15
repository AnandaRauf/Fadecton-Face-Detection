import face_recognition
import cv2
import numpy as np

print("--------------------------------------------\n")

namaprogram= "Fadecton(Face Detection)\n"
devby= "Developed by Ananda Rauf Maududi\n"
print(namaprogram)
print(devby)

print("--------------------------------------------\n")


kamera= cv2.VideoCapture(0);

image_people1 = face_recognition.load_image_file("rauf.jpg")
image_people1_encoding = face_recognition.face_encodings(image_people1)[0]
image_people2 = face_recognition.load_image_file("rhaina.jpg")
image_people2_encoding = face_recognition.face_encodings(image_people2)[0]
list_wajah_encoding = [
    image_people1_encoding,
    image_people1_encoding
]
list_nama_wajah_orang = [
    "Ananda Rauf Maududi"
    "Rhaina Kirana Arisahnti"
]
while(True):
    ret,gambar= kamera.read()
    rgb_frame = gambar[:, :, ::-1]
    lokasi_wajah = face_recognition.face_locations(rgb_frame)
    wajah_encoding = face_recognition.face_encodings(rgb_frame, lokasi_wajah)
    for (top, right, bottom, left), wajah_encoding in zip(lokasi_wajah, wajah_encoding):
        matches = face_recognition.compare_faces(list_wajah_encoding, wajah_encoding)
        oranglain = "Orang lain"
        jarak_wajah = face_recognition.face_distance(list_wajah_encoding, wajah_encoding)
        best_match_index = np.argmin(jarak_wajah)
        if matches[best_match_index]:
            oranglain = list_nama_wajah_orang[best_match_index]
            cv2.rectangle(gambar, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(gambar, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(gambar, oranglain, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        cv2.imshow('Fadecton(Face Detection)', gambar)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

kamera.release()
cv2.destroyAllWindows()
