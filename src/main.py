import cv2

def main():
    cap = cv2.VideoCapture(0)
    while(True):
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == 27:
            # To stop camera, press Escape key.
            break


if __name__ == "__main__":
    main()