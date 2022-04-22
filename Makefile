HOST=$(shell cat /etc/resolv.conf | grep nameserver | grep -oP '[\d.]+')

run:
	python darknet_integration --filter vehicle.audi.a2 --sync --host $(HOST)
run-async:
	python darknet_integration --filter vehicle.audi.a2 --host $(HOST)
run-linux:
	python darknet_integration --filter vehicle.audi.a2 --sync
