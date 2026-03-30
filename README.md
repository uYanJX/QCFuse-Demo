# Blend Test Run Guide

Sample test for attn blend. The paper is currently under review, and we will release additional components and a running guide in a future update.

## Run
```bash
python blend_test/sgblend.py --model_path /path/to/model --ratio 0.3 --max_new_tokens 32
```

Optional arguments example:

```bash
python blend_test/sgblend.py \
  --model_path /path/to/model \
  --ratio 0.3 \
  --max_new_tokens 32 \
  --attn_layer best_layer \
  --context_enhance
```
