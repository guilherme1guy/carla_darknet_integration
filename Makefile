HOST=$(shell ip addr show eth0 | grep -oP "(?<=inet\s)\d+(\.\d+){3}")

run:
	PYTHONPATH=./ python darknet_integration --filter vehicle.audi.a2 --host $(HOST)