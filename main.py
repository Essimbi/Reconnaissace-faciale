import cv2
import face_recognition

# Charger la base de données de visages connus
known_faces = []
known_names = []

a = 50
b = 50

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

for name in ["adrien", "deranot"]:
    # Charger l'image de chaque personne
    image = face_recognition.load_image_file("data/{}.jpeg".format(name))
    # Encoder le visage de chaque personne
    encoding = face_recognition.face_encodings(image)[0]
    # Ajouter l'encodage et le nom de la personne à la base de données
    known_faces.append(encoding)
    known_names.append(name)

# Initialiser la capture vidéo
video_capture = cv2.VideoCapture(0)

# Démarrer la boucle de traitement de la vidéo
while True:
    # Lire le cadre actuel de la vidéo
    _, frame = video_capture.read()

    # Encoder les visages dans le cadre actuel de la vidéo
    encodings = face_recognition.face_encodings(frame)

    # Convertissez l'image en niveaux de gris
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Pour chaque visage dans le cadre actuel de la vidéo
    for encoding in encodings:

        # Dessinez un rectangle autour de chaque visage détecté
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Comparer l'encodage avec chaque encodage de la base de données de visages connus
        matches = face_recognition.compare_faces(known_faces, encoding)

        # print(matches)

        # Si l'encodage correspond à un visage connu
        if True in matches:
            # Trouver l'index du visage connu le plus proche
            match_index = matches.index(True)
            # Récupérer le nom du visage connu le plus proche
            name = known_names[match_index]
        else:
            # Si aucun visage connu n'a été trouvé, définir le nom sur "Inconnu"
            name = "Inconnu"

        # Afficher le nom du visage sur l'image
        cv2.putText(frame, name, (a, b), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 255), 2)

    # Afficher l'image avec les noms des visages
    cv2.imshow("Video", frame)

    # Quitter la boucle si l'utilisateur appuie sur "q"
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Relâcher la capture vidéo et fermer toutes les fenêtres ouvertes
video_capture.release()
cv2.destroyAllWindows()
