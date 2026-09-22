import argparse
import sys

import cv2

points = []

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"Clicked: ({x}, {y})")

def main():
    parser = argparse.ArgumentParser(description="Click points on a frame to define a zone or line.")
    parser.add_argument("source", help="Video file or image path")
    parser.add_argument("--norm", action="store_true", help="Print normalized (0-1) coordinates")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"Failed to open {args.source}")
        return 1

    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Failed to read a frame.")
        return 1

    h, w = frame.shape[:2]
    
    cv2.namedWindow("pick_zone")
    cv2.setMouseCallback("pick_zone", mouse_callback)
    
    print("Click to add points. Press 'u' to undo. Press 'q' or Enter to finish.")
    while True:
        vis = frame.copy()
        if len(points) > 0:
            for i in range(len(points) - 1):
                cv2.line(vis, points[i], points[i+1], (0, 255, 0), 2)
            for p in points:
                cv2.circle(vis, p, 4, (0, 0, 255), -1)
        
        cv2.imshow("pick_zone", vis)
        key = cv2.waitKey(50) & 0xFF
        if key == ord('q') or key == 13:  # Enter
            break
        elif key == ord('u') and len(points) > 0:
            points.pop()
            print("Undo last point")

    cv2.destroyAllWindows()
    
    if not points:
        print("No points selected.")
        return 0

    print("\nYAML array:")
    print("[")
    for x, y in points:
        if args.norm:
            nx = round(x / w, 4)
            ny = round(y / h, 4)
            print(f"  [{nx}, {ny}],")
        else:
            print(f"  [{x}, {y}],")
    print("]")
    return 0

if __name__ == "__main__":
    sys.exit(main())
