llmd is a single binary that runs a hybrid system 1 + system 2 endpoint to be invoked from your code.

Basically the point of this is to be able to easily handle unstructured data in your code instead of writing custom fragile parsers.

It takes the concept of jev a bit further - you can still write traditional, hand-made, self-engineered code, but make your life a bit easier by handling over chunks of work to the LLM.

llmd handles decisions (think "is this email an ad or not"), it can parse text into JSON with the schema you provide, and it can produce text following a certain format.

llmd is meant to be run locally, so it wraps around laya multilingual and Qwen3 0.6B

The design goals:

- great ergonomics for the hobbyist developer
- self contained single binary, to be run in a container with your app, on your workstation, or on a constrained edge hardware
- low footprint, low RAM usage (sub 2GB RAM), low latency, high generation speed
- zero config required

The target platform this was developed for is an Intel N100 homelab/miniPC with no discrete GPU, a Raspberry Pi, etc.