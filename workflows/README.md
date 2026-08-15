Wan 2.2 ComfyUI workflows
=========================

These files are NOT working Wan workflows. They are documented adapter
TEMPLATES that describe the node structure the Wan provider expects and
validate against. They exist so the provider can:

  - validate the expected node classes are present
  - fail clearly with WORKFLOW_INVALID / WORKFLOW_NOT_FOUND
  - tell you exactly what must be exported from ComfyUI

To enable real generation
-------------------------

1. Start ComfyUI with the Wan 2.2 model weights installed (place weights in
   ComfyUI/models/checkpoints — do NOT put model weights in this repo).

2. In ComfyUI, open the Workflow menu -> Browse Templates -> Video ->
   "Wan2.2 5B video generation" (T2V) or the Wan2.2 I2V template.

3. Export as API FORMAT: Workflow menu -> Save (API Format).

4. REPLACE the contents of:
     workflows/wan22_t2v_api.json   (text-to-video)
     workflows/wan22_i2v_api.json   (image-to-video)
   with your exported API-format graph.

5. The provider validates that the exported graph contains the required node
   classes (loader, CLIPTextEncode for positive/negative, KSampler, VAEDecode,
   SaveVideo) and for I2V a LoadImage node. If a required node is missing the
   submission is rejected with WORKFLOW_INVALID.

Required node classes
---------------------

T2V (wan22_t2v_api.json):
  - loader: CheckpointLoaderSimple | UnetLoaderGGUF | WanModelLoader
  - positive_prompt: CLIPTextEncode
  - negative_prompt: CLIPTextEncode
  - sampler: KSampler
  - vae_decode: VAEDecode
  - save_video: SaveAnimatedWEBP | SaveVideo | VHS_VideoCombine

I2V (wan22_i2v_api.json): same as above PLUS
  - load_image: LoadImage

Model detection
---------------

The provider queries ComfyUI /object_info and looks for a model whose filename
contains "wan2.2". If none is found, generation fails with MODEL_NOT_FOUND and
reports the detected models + ComfyUI endpoint.

Never fake a workflow. Never fake model availability. Never fake an MP4.
