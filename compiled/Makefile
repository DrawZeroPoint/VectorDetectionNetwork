all:
	# install pycocotools locally
	python pycocotools/setup.py build_ext --inplace
	python keypoints/nms/setup.py build_ext --inplace
	rm -rf build
	rm -rf keypoints/nms/build
