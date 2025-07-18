## docker build
```
docker/build.sh
```
## docker run
```
docker/run.sh
```
## start REST server
```
source /catkin_ws/devel/setup.bash
cd /catkin_ws/src/irsl_instance_segmentation/rest_server
python3 app.py
```

## client test
```
docker exec -it irsl_instance_segmentation bash
```
```
cd /catkin_ws/src/irsl_instance_segmentation/rest_server 
wget http://images.cocodataset.org/val2017/000000439715.jpg -q -O input.jpg
python3 client_test.py --image_path input.jpg --server localhost --port 8008 --debug
```