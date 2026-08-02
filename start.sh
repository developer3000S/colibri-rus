MODEL_PATH="/disk/raid/models/glm5.2_i4"
docker run --rm -it --name colibri-c \
  -v "$MODEL_PATH:/app/glm-5.2" \
  -e COLI_MODEL=/app/glm-5.2 \
  colibri-i ./coli chat


# COLI_MODEL=/disk/raid/models/glm5.2_i4 /disk/raid/colibri-rus/c/coli chat     # RAM budget, cache and MTP auto-detected
# COLI_MODEL=/disk/raid/models/glm5.2_i4 /disk/raid/colibri-rus/c/coli plan     # inspect the planned VRAM/RAM/disk placement
# COLI_MODEL=/disk/raid/models/glm5.2_i4 /disk/raid/colibri-rus/c/coli doctor   # read-only readiness check
# COLI_MODEL=/disk/raid/models/glm5.2_i4 /disk/raid/colibri-rus/c/coli doctor --deep  # strict tensors/shards/index/mirror preflight
# COLI_MODEL=/disk/raid/models/glm5.2_i4 /disk/raid/colibri-rus/c/coli tune     # measure and save this machine's fastest safe execution profile
# /disk/raid/colibri-rus/c/coli web  --model /disk/raid/models/glm5.2_i4        # API + web dashboard on one port
# /disk/raid/colibri-rus/c/coli serve --model /disk/raid/models/glm5.2_i4       # OpenAI-compatible API only
