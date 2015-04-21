dai:
	cd pydeeplearn/image && python setup.py build_ext -i && cd ../..

clean:
	cd pydeeplearn/image && rm -rf build image.so image.pyd image.cpp && cd ../..
