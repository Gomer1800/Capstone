import Core.Camera.Subsystem as Camera
import cv2


# Press the green button in the gutter to run the script.
def test_webcam():
    camera = Camera.Subsystem(
        type="WEB",
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


def test_ipcam():
    camera = Camera.Subsystem(
        type="IP",
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


if __name__ == '__main__':
    test_webcam()
    test_ipcam()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
