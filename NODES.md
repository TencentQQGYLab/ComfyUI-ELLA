# Nodes reference

## Loaders

### Load ELLA Model

#### Configuration parameters
- **name**: (STRING), ELLA model to load.

#### Outputs
- **ella**: (ELLA), ELLA model.

### Load T5 TextEncoder #ELLA

#### Configuration parameters
- **name**: (STRING), T5 text encoder model to load.
- **max_length**: (INT), max_length of encoder, set `0` for [flexible token length](https://github.com/TencentQQGYLab/ELLA?tab=readme-ov-file#2-flexible-token-length).
- **dtype**: (BOOLEAN)
  
#### Outputs
- **t5_encoder**: (T5_TEXT_ENCODER)

## Main Apply Nodes

### Apply ELLA

#### Inputs
- **model**: (MODEL), loaded by `Load Checkpoint` and other model loaders.
- **ella**: (ELLA), ELLA model loaded by `Load ELLA Model` node.
- **positive**: (ELLA_EMBEDS), currently needing both
- **negative**:  (ELLA_EMBEDS), currently needing both

#### Outputs
- **model**: (MODEL), sd model with ELLA model injected.
- **positive**: (CONDITIONING), for `KSamplers`
- **negative**:  (CONDITIONING), for `KSamplers`

### T5 Text Encode #ELLA

#### Inputs
- **text**: (STRING), prompt to encode.
- **text_encoder**: (T5_TEXT_ENCODER)

#### Outputs
- **ella_embeds**: (ELLA_EMBEDS)

### Helpers

### ELLA Combine Embeds

#### Inputs
- **embeds**: (ELLA_EMBEDS)
- **embeds_add**: (ELLA_EMBEDS)

#### Outputs
- **embeds**: (ELLA_EMBEDS)

### Convert Condition to ELLA Embeds

use for convert clip `CONDITIONING` to `ELLA_EMBEDS`

#### Inputs
- **cond**: (CONDITIONING)

#### Outputs
- **embeds**: (ELLA_EMBEDS)
