MODEL_URL := https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt

checkpoint:
	mkdir -p checkpoint
	wget -P ./checkpoint/ ${MODEL_URL}


clean:
	rm -rf checkpoint

.PHONY: checkpoint
