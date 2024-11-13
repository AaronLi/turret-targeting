export HF_HOME = /mnt/16ddc835-5ca5-49c2-be87-083b2b0da576/Models/huggingface/

SHELL := /bin/bash

run:
	python main.py

run_amd:
	source ./amd_gpu.env && make run