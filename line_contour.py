import cv2
import numpy as np

cap = cv2.VideoCapture(0)   
cap.set(3, 160)  
cap.set(4, 120)  

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting.")
        break
    
    low_b = np.array([5, 5, 5], dtype=np.uint8)
    high_b = np.array([0, 0, 0], dtype=np.uint8)
    
    mask = cv2.inRange(frame, high_b, low_b)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        M = cv2.moments(c)
        
        if M["m00"] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            
            print(f"CX: {cx}, CY: {cy}")
            
            if cx >= 120:
                print("Turn Left")
            elif 40 < cx < 120:
                print("On Track!")
            else:  # cx <= 40
                print("Turn Right")
            
            cv2.circle(frame, (cx, cy), 5, (255, 255, 255), -1)
        
        cv2.drawContours(frame, [c], -1, (0, 255, 0), 1)
    else:
        print("I don't see the line")
    
    cv2.imshow("Mask", mask)
    cv2.imshow("Frame", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
