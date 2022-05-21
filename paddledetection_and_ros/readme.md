# 使用深度学习在gazebo仿真当中做检测   
1. 安装paddlepaddle和paddledetection，按照官方教程进行安装
2. cv_bridge使用python3编译。流程是下载cv_bridge源码，将CMakeLists.txt里的find_package(Boost REQUIRED python36)改为find_package(Boost REQUIRED python3)
3. 修改paddledetection推理源码。
4. 使用对应参数运行：<code> python3 /home/wangtao/PaddleDetection/deploy/python/infer.py --model_dir=/home/wangtao/icra_ws/src/line_maze_ros-master/src/picodet_s_320_voc --camera_id=0 --device=GPU </code>
