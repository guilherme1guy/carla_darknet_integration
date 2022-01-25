HOST=$(shell cat /etc/resolv.conf | grep nameserver | grep -oP '[\d.]+')

run:
	PYTHONPATH=./ python darknet_integration --filter vehicle.audi.a2 --host $(HOST)

run-linux:
	PYTHONPATH=./ python darknet_integration --filter vehicle.audi.a2