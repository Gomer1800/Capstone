import Core.Camera.Subsystem as Camera
import cv2

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    camera = Camera.Subsystem(
        type=None,
        name=None,
        camera_path=None,
        storage_path=None
    )
    camera.initialize()
    frame = camera.capture_image()
    while True:
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    camera.shutdown()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
