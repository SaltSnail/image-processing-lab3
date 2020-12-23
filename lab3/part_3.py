import cv2
import os.path as osp

from image_processing_lib.cli_image_argument import get_image_path
from image_processing_lib.windows_manager import create_two_windows
from image_processing_lib.time_comparing import get_time
from part_2_lib.circle_search import serch_circles, draw_circles
from part_3_lib.statistic import compare_circles, count_circles, histogram_circles


if __name__ == "__main__":
    image_path = get_image_path(
        default_path=osp.join(
            osp.dirname(__file__),
            "part_2_lib/src/HoughCircles.jpg",
        )
    )

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    blured_img = cv2.GaussianBlur(img, (5, 5), 0)
    edges = cv2.Canny(blured_img, 50, 150, apertureSize=3)

    minRc = 30
    maxRc = 60
    minRo = 10
    maxRo = 120

    circles_custom = get_time(serch_circles, edges, minRc, maxRc, 120, 40)
    circles_opencv = get_time(
        cv2.HoughCircles,
        edges,
        cv2.HOUGH_GRADIENT,
        1,
        5,
        param1=120,
        param2=40,
        minRadius=minRo,
        maxRadius=maxRo,
    )

    my_detected_circles = draw_circles(circles_custom, img)
    opencv_detected_circles = draw_circles(circles_opencv[0], img)

    create_two_windows(
        opencv_detected_circles,
        my_detected_circles,
        'opencv_detected_circles',
        'my_detected_circles',
    )

    print("\nOpencv as reference")
    compare_results = compare_circles(img.shape,
                                      ref_circles=circles_opencv[0],
                                      actual_circles=circles_custom)

    print("\nCount opencv circles: ", count_circles(circles_opencv[0]))
    print("Count custom circles: ", count_circles(circles_custom))
    print("\nHistograms are using total found circles")
    print("Total found opencv circles: ", len(circles_opencv[0]))
    print("Total found custom circles: ", len(circles_custom))

    print("Custom histogram circle radius min =", minRc, "max =", maxRc)
    print("Opencv histogram circle radius min =", minRo, "max =", maxRo)
    hist_custom = histogram_circles(circles_custom, minRc, maxRc)
    hist_opencv = histogram_circles(circles_opencv[0], minRo, maxRo)
    create_two_windows(
        hist_custom,
        hist_opencv,
        'histogram custom',
        'histogram opencv',
    )
