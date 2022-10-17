#
# Load an image (from its image_file_name) and show it in opencv's window...
#

import cv2

image_file_name = "coffee.jpg"

#img = cv2.imread(image_file_name,0)  # b+w
img = cv2.imread(image_file_name)  # color 
cv2.imshow('image',img)   # show image in separate window
cv2.waitKey(0)            # wait for a keypress... an opencv idiosyncrasy!
cv2.destroyAllWindows()   # this doesn't seem to _actually_ close the window,
                          # but closing the python prompt does... 
    


