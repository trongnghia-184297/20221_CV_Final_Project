import cv2
import numpy as np
  
class count_connected_comp:
    def __init__(self) -> None:
        self.origin_img = None
        self.img = None
        self.output = None
        self.rec_img = None

    def get_shape(self, img):
        height = img.shape[0]
        width = img.shape[1]
        channels = img.shape[2]

        return height, width

    def visualization_window(self, pos, size ,img, name):
        img = cv2.resize(img, size) 
        cv2.imshow(name, img)
        cv2.moveWindow(name, pos[0], pos[1])

    def Counting(self,img_source:str,img_origin:str):
        # Config step = 3
        size3 = (540,540)
        pos30 = (70,250)
        pos31 = (690,250)
        pos32 = (1310,250)


        self.img_origin = cv2.imread(img_origin)
        self.visualization_window(pos30, size3, self.img_origin, "Step1: Visual original image")

        # Loading the image
        # img = cv2.imread('project1/denoise/noise_periodic.png')
        # img =cv2.imread('project1/denoise/noise_pepper.png')
        self.img = cv2.imread(img_source)
        img = self.img
        self.visualization_window(pos31, size3, img, "Step2: Denoise image")
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
        # self.visualization_window(pos31, size3, threshold, "Step2: Adaptive thresholding")


        #apply opening
        kernelSize = (4,4) 
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize)
        # print(kernel)
        opening_img = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)
        self.visualization_window(pos32, size3, opening_img, "Step3: Adaptive threshold + Opening operation")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # cv2.imshow("threshold",opening_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()  

        # Apply the Component analysis function
        analysis = cv2.connectedComponentsWithStats(opening_img,
                                                    4,
                                                    cv2.CV_32S)
        (totalLabels, label_ids, values, centroid) = analysis

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

        # Visualization
        rec_img = img.copy()
        out_img = self.img_origin.copy()
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
                cv2.rectangle(out_img,pt1,pt2,
                            (0, 255, 0), 3)
                cv2.circle(out_img, (int(X),
                                    int(Y)), 4, (0, 0, 255), -1)
        
                # Create a new array to show individual component
                component = np.zeros(gray_img.shape, dtype="uint8")
                componentMask = (label_ids == i).astype("uint8") * 255
        
                # Apply the mask using the bitwise operator
                component = cv2.bitwise_or(component,componentMask)
                output = cv2.bitwise_or(output, componentMask)
                
                if i < 8 or i > 90:
                    # Show the final images
                    self.visualization_window(pos30, size3, new_img, "Step4-Counting: Image")
                    self.visualization_window(pos31, size3, component, "Step4-Counting: Individual Component")
                    self.visualization_window(pos32, size3, output, "Step4-Counting: Total Component")
                    # cv2.imshow("Image", new_img)
                    # cv2.imshow("Individual Component", component)
                    # cv2.imshow("Filtered Components", output)
                    cv2.waitKey(0)
        # cv2.imshow("Image", new_img)
        # Final result
        size0 = (640,640)
        pos0 = (600,200)

        # Visualize text
        h_img, w_img = self.get_shape(img)
        font = cv2.FONT_HERSHEY_SIMPLEX
        position_1 = (w_img-40, h_img-25)
        position_2 = (w_img-35, h_img-10)
        fontScale  = 0.5
        fontColor  = (0,0,255)
        thickness  = 2

        cv2.putText(out_img,'Total', position_1, font, fontScale, fontColor, thickness)
        cv2.putText(out_img,str(count), position_2, font, fontScale, fontColor, thickness)

        self.visualization_window(pos0, size0, out_img, "Step5: Visualization final output")
        # cv2.imshow("Filtered Components", rec_img)

        cv2.waitKey(0)
        cv2.destroyAllWindows()   

        
        self.output=output
        self.rec_img=rec_img
        return count
  
# count_connected_comp = count_connected_comp()
# res = count_connected_comp.Counting('project1/denoise/noise_periodic.png')
# cv2.imshow("Image", count_connected_comp.img)
# cv2.imshow("Filtered Components", count_connected_comp.output)
# cv2.imshow("Rec img", count_connected_comp.rec_img)
# # cv2.imshow("threshold",opening_img)
# print(res)
# cv2.waitKey(0)
# cv2.destroyAllWindows() 