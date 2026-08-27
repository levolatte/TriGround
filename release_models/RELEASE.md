# Publishing the model assets

The two `.pt` files in `release_models/v1/` are intentionally ignored by Git.
Publish them as assets of the same GitHub Release after pushing the code branch.

Recommended release metadata:

```text
Tag: models-v1.0.0
Title: TriGround representative checkpoints v1
```

With GitHub CLI installed and authenticated:

```bash
gh release create models-v1.0.0 \
  release_models/v1/triground-rdt-ws-v1-manual-ft1.pt \
  release_models/v1/triground-parallel-a-v1.pt \
  release_models/v1/SHA256SUMS \
  release_models/v1/release-manifest.json \
  --title "TriGround representative checkpoints v1" \
  --notes-file release_models/MODEL_CARD.md
```

The same layout can be created in the GitHub web interface under
**Releases → Draft a new release**. Do not force-add the `.pt` files to the Git
repository. Verify downloaded files against `SHA256SUMS`.

