# Smart Cashier System 
This project was implemented by following tutorials from [Python Lessons](pylessons.com). For the program to work, it requires the user to:
1. Create a database: `create_facedatabase.py`
   - To look inside the database, use [DB Browser for SQLite](https://sqlitebrowser.org)
2. Run `face_enrollment.py` to capture the user's face with a user ID and initial balance of 100 to be stored into the database. The images are stored in `/custom_dataset/`.
3. Train the program to recognize the user's face by running `face_enroll_trainer.py`. It will then export a .yml file to `/recognizer/`.
4. Preparing the Image dataset for Custom Object Detection
   - Use [LabelImg](github.com/tzutalin/labelimg)
      - `windows_v1.2/labelImg.exe`
      - LabelImg is a graphical image annotation tool. The annotations are saved as XML files in PASCAL VOC format and supports YOLO format. 
      - This is very time consuming as it requires the user to download hundreds of images and to specify where the custom object is in each of the images. 
   - Use [OIDv4_ToolKit](https://github.com/EscVM/OIDv4_ToolKit)
      - This toolkit is capable of downloading hundreds of images from [Google Open Images V6](https://storage.googleapis.com/openimages/web/index.html).
      - The toolkit will ask you to download **class-descriptions-boxable.csv** (contains the name of hundreds of classes with their corresponding `LabelName` file), **test-annotations-bbox.csv** and **train-annotations-bbox.csv** (bbox: bounding box is a rectangular box surrounding the object by having coordinates from the .csv files to describe the object location).
      - To download images such as the one used in this project: 
         - Training images: `python main.py downloader --classes Apple Banana Orange Bagel Croissant --type_csv train --limit 300`
         - Testing images: `python main.py downloader --classes Apple Banana Orange Bagel Croissant --type_csv test --limit 100`
         - The images are located in `OIDv4_ToolKit/OID/` with the csv and dataset folder. The dataset folder should have the 'train' and 'test' folder.
      - Next, convert the label files to XML by going into `tools/` and run the `oid_to_pascal_voc_xml.py` script. 
      - Finally, convert the XML file to YOLO file structure from the same folder `tools/` and run the `XML_to_YOLOv3.py` script. 
      - Once it finishes the conversion, the files: `Dataset_names.txt`, `Dataset_train.txt`, and `Dataset_test.txt` will be located in the `model_data` folder.
5. To train the program, configure the `configs.py` in `yolov3/`.
    - Change the `TRAIN_CLASSES` to `"./model_data/Dataset_names.txt"` and `TRAIN_ANNOT_PATH` to `"./model_data/Dataset_train.txt"`.
    - Change the `TRAIN_ANNOT_PATH` to `"./model_data/Dataset_test.txt"`.
    - The `BATCH_SIZE` was halved to prevent a training error on my PC. 
6. Run `python train.py` to train the images.
    - To look at the training process: `tensorboard --logdir=log`. 
    - To look at the tensorboard again, e.g. if you close the program and want to re-check it: `tensorboard --logdir=./log`
7. Run `main.py` to now detect the face and objects using the webcam. 
   - Showing one of the image class from the OIDv4 ToolKit into the webcam will subtract 1 from the user's balance.  

