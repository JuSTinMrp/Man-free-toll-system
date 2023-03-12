# Man-free-toll-system

To mitigate the time in the tollgates especially during festivals, we are introducing a man free toll collection system by just slowing down the vehicles using speed breakers.
It is a IoT  based number plate detection. This is based on the python. 
Our device(consists of Raspberry pi) will detect the number plate using pi camera (OV5647 chipset)and collects the amount from the database(which links with the car owner account) and sends a sms to the owner. In case of no amount in their acc, value will get negative which sense they need to pay fine for it within particular time.
This is very useful for the logistics companies to reduce time delay while delivering products.
This innovative idea introduced here is to reduce the delivery delays of logistics companies.


We are going to use modules like Opencv, Pandas, Tesseract in our python code to detect the license plate and then also entering the vehicles entry and exit in a csv file(Excel sheet).
It collects the toll amount from the database which links with the car owner.
Image Processing includes resizing, Canny edgeding, Gray-scale, Denoising.


