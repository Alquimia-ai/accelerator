mkfile_path := $(abspath $(lastword $(MAKEFILE_LIST)))
mkfile_dir := $(dir $(mkfile_path))

REGISTRY ?= docker.io/alquimiaai
TAG ?= latest
ENGINE ?= docker
PLATFORM ?= linux/amd64
DEPLOY_NAMESPACE ?= alquimia-runtime

.PHONY: build
build:
	echo 'Building base image';
	$(ENGINE) build -t $(REGISTRY)/technical-content-generator-mcp:${TAG} -f $(mkfile_dir)/Dockerfile . --platform=$(PLATFORM)

.PHONY: push
push:
	@echo "Pushing images to $(REGISTRY)..."
	$(ENGINE) push $(REGISTRY)/technical-content-generator-mcp:${TAG}

.PHONY: recreate
recreate: uninstall deploy

.PHONY: uninstall
uninstall:
	@oc delete -f $(mkfile_dir)/deploy.yaml

.PHONY: deploy
deploy:
	@oc apply -f $(mkfile_dir)/deploy.yaml -n $(DEPLOY_NAMESPACE)
	@watch -n1 oc get pods -n $(DEPLOY_NAMESPACE)

.PHONY: package
package:
	@uv build
