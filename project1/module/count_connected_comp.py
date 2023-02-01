import cv2
import numpy as np
  
class count_connected_comp:
    def __init__(self) -> None:
        self.img = None
        self.output = None
        self.rec_img = None
    def Counting(self,img_source:str):
        # Loading the image
        # img = cv2.imread('project1/denoise/noise_periodic.png')
        # img =cv2.imread('project1/denoise/noise_pepper.png')
        self.img = cv2.imread(img_source)
        img = self.img
        # preprocess the image
        gray_img = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
        
        # Applying 7x7 Gaussian Blur
        blurred = cv2.GaussianBlur(gray_img, (5,5), 0)
        
        # Applying threshold
        # threshold = cv2.threshold(blurred, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1] 
        thresh_adaptive = 180
        kernelSize = 25
        C = -10
        threshold = cv2.adaptiveThreshold(blurred,thresh_adaptive,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,kernelSize, C)

        #apply opening
        kernelSize = (7, 9) 
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize)
        # print(kernel)
        opening_img = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)
        # cv2.imshow("threshold",opening_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()  

        # Apply the Component analysis function
        analysis = cv2.connectedComponentsWithStats(opening_img,
                                                    4,
                                                    cv2.CV_32S)
        (totalLabels, label_ids, values, centroid) = analysis
        print(len(values[0]))
        print(label_ids)
        # Initialize a new image to store 
        # all the output components
        output = np.zeros(gray_img.shape, dtype="uint8") 

        #avg area 
        sum_area=0
        for i in range(1, totalLabels):
            # Area of the component
            area = values[i, cv2.CC_STAT_AREA] 
            sum_area += area
        avg_area = float(sum_area/totalLabels)
        rec_img = img.copy()
        count=0
        # Loop through each component
        for i in range(1, totalLabels):
            
            # Area of the component
            area = values[i, cv2.CC_STAT_AREA] 
            
            if area/avg_area > 0.18:
                # componentMask = (label_ids == i).astype("uint8") * 255
                # output = cv2.bitwise_or(output, componentMask)
                new_img=img.copy()
                count +=1
                # Now extract the coordinate points
                x1 = values[i, cv2.CC_STAT_LEFT]
                y1 = values[i, cv2.CC_STAT_TOP]
                w = values[i, cv2.CC_STAT_WIDTH]
                h = values[i, cv2.CC_STAT_HEIGHT]
                
                # Coordinate of the bounding box
                pt1 = (x1, y1)
                pt2 = (x1+ w, y1+ h)
                (X, Y) = centroid[i]
                
                # Bounding boxes for each component
                cv2.rectangle(new_img,pt1,pt2,
                            (0, 255, 0), 3)
                cv2.circle(new_img, (int(X),
                                    int(Y)), 
                        4, (0, 0, 255), -1)
                cv2.rectangle(rec_img,pt1,pt2,
                            (0, 255, 0), 3)
                cv2.circle(rec_img, (int(X),
                                    int(Y)), 
                        4, (0, 0, 255), -1)
        
                # Create a new array to show individual component
                component = np.zeros(gray_img.shape, dtype="uint8")
                componentMask = (label_ids == i).astype("uint8") * 255
        
                # Apply the mask using the bitwise operator
                component = cv2.bitwise_or(component,componentMask)
                output = cv2.bitwise_or(output, componentMask)
                
                # Show the final images
                # cv2.imshow("Image", new_img)
                # cv2.imshow("Individual Component", component)
                # cv2.imshow("Filtered Components", output)
                # cv2.waitKey(0)
        
        self.output=output
        self.rec_img=rec_img
        return count
  
count_connected_comp = count_connected_comp()
res = count_connected_comp.Counting('project1/denoise/noise_periodic.png')
cv2.imshow("Image", count_connected_comp.img)
cv2.imshow("Filtered Components", count_connected_comp.output)
cv2.imshow("Rec img", count_connected_comp.rec_img)
# cv2.imshow("threshold",opening_img)
print(res)
cv2.waitKey(0)
# cv2.destroyAllWindows() 