#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from geometry_msgs.msg import Vector3
import math

def normalize_angle(angle):
    """
    Normalize angle to the range [-180, 180] degrees.
    """
    while angle > 180:
        angle -= 360
    while angle < -180:
        angle += 360
    return angle

class LineFollowerNode:
    def __init__(self):
        rospy.init_node('line_follower', anonymous=True)

        # Subscribers
        rospy.Subscriber('/usb_cam_node/image_raw', Image, self.camera_callback)
        rospy.Subscriber('/sensors/orientation', Vector3, self.imu_callback)

        # Publishers
        self.angle_pub = rospy.Publisher('/line/angle_difference', Float32, queue_size=10)
        self.image_pub = rospy.Publisher('/processed_image', Image, queue_size=10)

        # Variables
        self.yaw = 0.0  # IMU yaw
        self.bridge = CvBridge()

    def imu_callback(self, msg):
        """
        Callback to handle IMU data and update yaw.
        """
        self.yaw = msg.z  # Assuming `orientation.z` is in degrees (0 to 360)

    def camera_callback(self, msg):
        """
        Callback to process camera image, detect line, and calculate angle difference.
        """
        try:
            # Convert ROS image to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
            return

        # Preprocess the image
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV)

        # Detect line using contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Find the largest contour (assuming it's the line)
            largest_contour = max(contours, key=cv2.contourArea)

            # Fit a line to the contour
            [vx, vy, x, y] = cv2.fitLine(largest_contour, cv2.DIST_L2, 0, 0.01, 0.01)
            line_angle = math.degrees(math.atan2(vy, vx))

            # Normalize line angle to [0, 360]
            if line_angle < 0:
                line_angle += 360

            # Calculate angle difference
            angle_difference = normalize_angle(line_angle - self.yaw)

            # Publish the angle difference
            rospy.loginfo(f"Angle Difference: {angle_difference:.2f} degrees")
            self.angle_pub.publish(angle_difference)

            # Draw the line and yaw axis on the image
            rows, cols = cv_image.shape[:2]
            lefty = int((-x * vy / vx) + y)
            righty = int(((cols - x) * vy / vx) + y)
            cv2.line(cv_image, (cols - 1, righty), (0, lefty), (0, 255, 0), 2)

            # Draw yaw axis
            yaw_x = int(cols // 2 + 100 * math.cos(math.radians(self.yaw)))
            yaw_y = int(rows // 2 - 100 * math.sin(math.radians(self.yaw)))
            cv2.line(cv_image, (cols // 2, rows // 2), (yaw_x, yaw_y), (0, 0, 255), 2)
            cv2.putText(cv_image, f"Yaw: {self.yaw:.1f} deg", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            # Debug: Check if the processed image is valid
            rospy.loginfo("Publishing processed image.")

            # Publish the processed image
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))

            # Optional: Display for debugging purposes
            cv2.imshow("Line Detection", cv_image)
            cv2.waitKey(1)
        else:
            rospy.logwarn("No line detected in the frame.")

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        node = LineFollowerNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()
