

- convert2Yolo
- val2017 json to txt

~~~
$ python3 example.py --datasets COCO --img_path ~/Downloads/coco/val2017/ --label /home/djjin/Downloads/coco/annotations_trainval2017/annotations/instances_val2017.json --convert_output_path /home/djjin/Downloads/coco/output/ --img_type ".jpg" --manifest_path ./ --cls_list_file /home/djjin/Downloads/coco/coco.names 
~~~

- train json to txt

~~~
$ python3 example.py --datasets COCO --img_path ~/Downloads/coco/train2017/ --label /home/djjin/Downloads/coco/annotations_trainval2017/annotations/instances_train2017.json --convert_output_path /home/djjin/Downloads/coco/train_output/ --img_type ".jpg" --manifest_path ./ --cls_list_file /home/djjin/Downloads/coco/coco.names 
~~~

- make AP

~~~
./darknet detector valid /home/djjin/Downloads/coco/coco.data /home/djjin/Downloads/coco/yolov4.cfg yolov4.weights
~~~

- train

~~~
./darknet detector train /home/djjin/Downloads/coco/coco.data /home/djjin/Downloads/coco/yolov4.cfg yolov4.conv.137
~~~

