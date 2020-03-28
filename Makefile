dev:
	docker run --rm -ti --volume $(shell pwd):/app wekaco/$(shell basename $(shell pwd) | tr "[:upper:]" "[:lower:]"):devel

build:
	docker build -t wekaco/$(shell basename $(shell pwd) | tr "[:upper:]" "[:lower:]"):devel -f Dockerfile.devel .
