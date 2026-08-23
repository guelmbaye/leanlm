# Prompt templates

The grounded template lives in `packages/inference/services.py` rather than in a
file that can be edited without a version bump. A prompt is not configuration:
changing it changes what the model does, and every measurement taken before the
change becomes incomparable to every measurement taken after it.

What is recorded on every run instead:

- `template_id` — `leanlm.enterprise.grounded`
- `template_version` — bumped whenever the wording changes
- the rendered prompt's checksum, in the `PromptPackage` artifact

The files here document each version so that an old measurement can be read
alongside the prompt that produced it.
